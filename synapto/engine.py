import os
import gc
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from safetensors.torch import save_file, load_file


class SecurityError(Exception):
    """Исключение при нарушении векторов безопасности библиотеки."""
    pass


class SafetyUtils:
    """
    Модуль валидации и защиты данных.
    Предотвращает Path Traversal, DoS по памяти и некорректные типы.
    """
    @staticmethod
    def validate_and_sanitize_path(filepath: str) -> Path:
        if not isinstance(filepath, str) or not filepath.strip():
            raise SecurityError("Путь к файлу должен быть непустой строкой.")
            
        path = Path(filepath).resolve()
        if path.suffix != ".safetensors":
            raise SecurityError("Разрешена работа только с безопасным форматом .safetensors.")
            
        return path

    @staticmethod
    def sanitize_input_text(text: str, max_chars: int = 4096) -> str:
        if not isinstance(text, str):
            raise TypeError("Входные данные должны быть строкового типа.")
        if len(text) > max_chars:
            return text[:max_chars]
        return text


class PlasticityController:
    """
    Управляет параметром пластичности P [-1.0 .. 2.0].
    """
    def __init__(self, p_value: float = 1.0, base_lr: float = 2e-5):
        self.base_lr = base_lr
        self.set_plasticity(p_value)

    def set_plasticity(self, p_value: float) -> None:
        self.p_value = max(-1.0, min(2.0, float(p_value)))
        if self.p_value <= -1.0:
            self.lr = 0.0
            self.threshold = float('inf')
            self.is_frozen = True
        else:
            self.lr = self.base_lr * (self.p_value + 1.0)
            self.threshold = 1.0 / (1.0 + max(0.0, self.p_value))
            self.is_frozen = False


class DynamicModelWrapper(nn.Module):
    """
    Обертка над гибридной моделью: 4-bit квантованное статическое ядро + FP16 динамические слои.
    """
    def __init__(self, base_model: nn.Module, dynamic_layers_count: int = 4):
        super().__init__()
        self.model = base_model
        
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise AttributeError("Неподдерживаемая архитектура модели. Ожидается структура CausalLM (Llama/Qwen).")

        total_layers = len(self.model.model.layers)
        if dynamic_layers_count >= total_layers or dynamic_layers_count <= 0:
            raise ValueError(f"Количество динамических слоев должно быть в диапазоне от 1 до {total_layers - 1}.")

        static_layers_count = total_layers - dynamic_layers_count

        for param in self.model.parameters():
            param.requires_grad = False

        self.dynamic_params: List[nn.Parameter] = []
        for i in range(static_layers_count, total_layers):
            layer = self.model.model.layers[i]
            for param in layer.parameters():
                param.requires_grad = True
                self.dynamic_params.append(param)

        if not self.dynamic_params:
            raise RuntimeError("Не удалось выделить параметры для динамической памяти.")

        self.optimizer = torch.optim.AdamW(self.dynamic_params, lr=2e-5, weight_decay=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids).logits


class SynaptoEngine:
    """
    Основной движок Synaptic Weight Eviction (SWE) с ведением журнала консолидированной памяти.
    """
    def __init__(
        self, 
        model_id: str, 
        p_value: float = 1.5, 
        dynamic_layers: int = 4,
        max_replay_buffer_size: int = 50
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plasticity = PlasticityController(p_value=p_value)
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer: List[Tuple[str, str]] = []
        self.memory_journal: List[Dict[str, Any]] = []

        model_config = AutoConfig.from_pretrained(model_id)
        total_layers = getattr(model_config, "num_hidden_layers", 28)

        skip_layers = [f"model.layers.{i}" for i in range(total_layers - dynamic_layers, total_layers)]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_layers
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if self.device.type == "cuda" else None
        )

        self.wrapper = DynamicModelWrapper(base_model, dynamic_layers_count=dynamic_layers)

    def _calculate_masked_loss(self, prompt_text: str, completion_text: str) -> torch.Tensor:
        full_text = prompt_text + completion_text
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) <= len(prompt_ids):
            raise ValueError("Completion текст не содержит валидных токенов после промпта.")

        input_ids = torch.tensor([full_ids[:-1]], device=self.device)
        target_ids = torch.tensor([full_ids[1:]], device=self.device).clone()

        prompt_len = len(prompt_ids)
        target_ids[0, :prompt_len - 1] = -100

        logits = self.wrapper(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_ids.view(-1), 
            ignore_index=-100
        )
        return loss

    def consolidate(self, prompt_text: str, completion_text: str) -> bool:
        if self.plasticity.is_frozen or self.plasticity.lr == 0.0:
            return False

        prompt_text = SafetyUtils.sanitize_input_text(prompt_text)
        completion_text = SafetyUtils.sanitize_input_text(completion_text)

        try:
            self.wrapper.model.eval()
            with torch.no_grad():
                new_loss = self._calculate_masked_loss(prompt_text, completion_text)
                surprise_score = new_loss.item()

            if torch.isnan(new_loss) or torch.isinf(new_loss):
                return False

            if surprise_score > self.plasticity.threshold:
                for param_group in self.wrapper.optimizer.param_groups:
                    param_group['lr'] = self.plasticity.lr

                self.wrapper.model.train()
                total_loss = self._calculate_masked_loss(prompt_text, completion_text)

                if self.replay_buffer:
                    replay_samples = random.sample(
                        self.replay_buffer, 
                        min(2, len(self.replay_buffer))
                    )
                    for r_prompt, r_comp in replay_samples:
                        total_loss = total_loss + (0.5 * self._calculate_masked_loss(r_prompt, r_comp))

                self.wrapper.optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.wrapper.dynamic_params, max_norm=0.5)
                self.wrapper.optimizer.step()

                pair = (prompt_text, completion_text)
                if pair not in self.replay_buffer:
                    if len(self.replay_buffer) >= self.max_replay_buffer_size:
                        self.replay_buffer.pop(0)
                    self.replay_buffer.append(pair)
                    
                    self.memory_journal.append({
                        "prompt": prompt_text,
                        "completion": completion_text,
                        "surprise_score": surprise_score
                    })

                return True
            return False

        except torch.cuda.OutOfMemoryError:
            self.wrapper.optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            return False

    def generate_response(self, prompt_text: str, max_new_tokens: int = 32) -> str:
        prompt_text = SafetyUtils.sanitize_input_text(prompt_text)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        self.wrapper.model.eval()
        with torch.no_grad():
            output = self.wrapper.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_memory_dump(self) -> List[Dict[str, Any]]:
        """
        Возвращает реестр всех фактически зафиксированных в весах фрагментов памяти.
        """
        return self.memory_journal

    def save_memory_profile(self, filepath: str) -> bool:
        validated_path = SafetyUtils.validate_and_sanitize_path(filepath)
        state_dict: Dict[str, torch.Tensor] = {}
        for name, param in self.wrapper.model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.detach().cpu()

        if not state_dict:
            raise SecurityError("Нет доступных динамических весов.")

        save_file(state_dict, str(validated_path))
        return True

    def load_memory_profile(self, filepath: str) -> bool:
        validated_path = SafetyUtils.validate_and_sanitize_path(filepath)
        if not validated_path.exists():
            raise FileNotFoundError(f"Файл {validated_path} не найден.")

        loaded_tensors = load_file(str(validated_path))
        model_state = self.wrapper.model.state_dict()

        for name, tensor in loaded_tensors.items():
            if name in model_state and model_state[name].requires_grad:
                if model_state[name].shape != tensor.shape:
                    raise ValueError("Размерность тензора не совпадает.")
                model_state[name].copy_(tensor.to(self.device))

        return True


class ChatStreamProcessor:
    """
    Модуль авто-управления чатом. 
    Автоматически консолидирует выпадающие из KV-кэша реплики в веса.
    """
    def __init__(self, engine: SynaptoEngine, max_window_tokens: int = 512):
        self.engine = engine
        self.max_window_tokens = max_window_tokens
        self.chat_history: List[Tuple[str, str]] = []

    def process_turn(self, user_prompt: str, assistant_completion: str) -> None:
        self.chat_history.append((user_prompt, assistant_completion))
        
        total_tokens = sum(
            len(self.engine.tokenizer.encode(p + c)) 
            for p, c in self.chat_history
        )

        while total_tokens > self.max_window_tokens and len(self.chat_history) > 1:
            evicted_prompt, evicted_completion = self.chat_history.pop(0)
            self.engine.consolidate(evicted_prompt, evicted_completion)

            total_tokens = sum(
                len(self.engine.tokenizer.encode(p + c)) 
                for p, c in self.chat_history
            )