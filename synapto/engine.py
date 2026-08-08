import os
import gc
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from safetensors.torch import save_file, load_file

from .security import SafetyUtils, SecurityError, CryptoVault
from .plasticity import PlasticityController
from .wrapper import DynamicModelWrapper
from .queue import ConsolidationQueue


class SynaptoEngine:
    def __init__(
        self, 
        model_id: str, 
        p_value: float = 1.5, 
        dynamic_layers: int = 4,
        max_replay_buffer_size: int = 50,
        anchor_lambda: float = 1e-4
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plasticity = PlasticityController(p_value=p_value)
        self.max_replay_buffer_size = max_replay_buffer_size
        self.anchor_lambda = anchor_lambda
        self.replay_buffer: List[Tuple[str, str]] = []
        self.memory_journal: List[Dict[str, Any]] = []
        self.write_queue = ConsolidationQueue()

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
        """
        Точный расчет Target Loss Masking с учётом ChatML токенов ассистента.
        """
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": completion_text}
            ]
            full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(self.device)
            
            # Точный расчет длины промпта с включением тегов ассистента
            prompt_only = [messages[0]]
            prompt_ids = self.tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            prompt_len = prompt_ids.shape[1]
        else:
            full_text = prompt_text + completion_text
            prompt_ids_vec = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids_vec = self.tokenizer.encode(full_text, add_special_tokens=False)
            full_ids = torch.tensor([full_ids_vec], device=self.device)
            prompt_len = len(prompt_ids_vec)

        if full_ids.shape[1] <= prompt_len:
            raise ValueError("Completion текст не содержит валидных токенов.")

        input_ids = full_ids[:, :-1]
        target_ids = full_ids[:, 1:].clone()

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
                        min(3, len(self.replay_buffer))
                    )
                    replay_loss = torch.tensor(0.0, device=self.device)
                    for r_prompt, r_comp in replay_samples:
                        replay_loss = replay_loss + self._calculate_masked_loss(r_prompt, r_comp)
                    total_loss = total_loss + (0.4 * (replay_loss / len(replay_samples)))

                anchor_penalty = self.wrapper.calculate_anchor_loss(lambda_anchor=self.anchor_lambda)
                total_loss = total_loss + anchor_penalty

                self.wrapper.optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.wrapper.dynamic_params, max_norm=0.3)
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

    def enqueue_fact(self, prompt_text: str, completion_text: str) -> bool:
        return self.write_queue.push(prompt_text, completion_text)

    def process_queue(self, batch_size: int = 1) -> int:
        batch = self.write_queue.pop_batch(batch_size=batch_size)
        consolidated_count = 0
        for prompt_text, completion_text in batch:
            if self.consolidate(prompt_text, completion_text):
                consolidated_count += 1
        return consolidated_count

    def generate_response(self, prompt_text: str, max_new_tokens: int = 32) -> str:
        prompt_text = SafetyUtils.sanitize_input_text(prompt_text)
        
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt_text

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        self.wrapper.model.eval()
        with torch.no_grad():
            output = self.wrapper.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_memory_dump(self) -> List[Dict[str, Any]]:
        return self.memory_journal

    def reset_memory(self) -> None:
        self.wrapper.reset_memory_weights()
        self.replay_buffer.clear()
        self.memory_journal.clear()
        self.write_queue.queue.clear()

    def save_memory_profile(self, filepath: str, encryption_key: Optional[str] = None) -> bool:
        validated_path = SafetyUtils.validate_and_sanitize_path(filepath)
        
        state_dict: Dict[str, torch.Tensor] = {}
        for name, param in self.wrapper.model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.detach().cpu()

        if not state_dict:
            raise SecurityError("Нет доступных динамических весов.")

        save_file(state_dict, str(validated_path))

        metadata = {
            "replay_buffer": self.replay_buffer,
            "memory_journal": self.memory_journal
        }

        if encryption_key:
            meta_path = validated_path.with_suffix(".enc")
            encrypted_bytes = CryptoVault.encrypt_metadata(metadata, encryption_key)
            with open(meta_path, "wb") as f:
                f.write(encrypted_bytes)
        else:
            meta_path = validated_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    def load_memory_profile(self, filepath: str, encryption_key: Optional[str] = None) -> bool:
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

        if encryption_key:
            meta_path = validated_path.with_suffix(".enc")
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    encrypted_bytes = f.read()
                metadata = CryptoVault.decrypt_metadata(encrypted_bytes, encryption_key)
                self.replay_buffer = metadata.get("replay_buffer", [])
                self.memory_journal = metadata.get("memory_journal", [])
        else:
            meta_path = validated_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self.replay_buffer = metadata.get("replay_buffer", [])
                self.memory_journal = metadata.get("memory_journal", [])

        return True