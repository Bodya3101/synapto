from typing import List
import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    HAS_BNB_OPTIM = True
except ImportError:
    HAS_BNB_OPTIM = False


class DynamicModelWrapper(nn.Module):
    """
    Обертка над моделью: 4-bit база + FP16 динамические слои.
    """
    def __init__(self, base_model: nn.Module, dynamic_layers_count: int = 4):
        super().__init__()
        self.model = base_model
        
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise AttributeError("Неподдерживаемая архитектура модели. Ожидается структура CausalLM (Llama/Qwen).")

        total_layers = len(self.model.model.layers)
        if dynamic_layers_count >= total_layers or dynamic_layers_count <= 0:
            raise ValueError(f"Количество динамических слоев должно быть от 1 до {total_layers - 1}.")

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

        # Снимок базовых весов для Elastic Weight Anchoring и отката
        self.initial_anchor_weights = [param.detach().clone() for param in self.dynamic_params]

        if HAS_BNB_OPTIM:
            self.optimizer = bnb.optim.AdamW8bit(self.dynamic_params, lr=2e-5, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.AdamW(self.dynamic_params, lr=2e-5, weight_decay=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids).logits

    def calculate_anchor_loss(self, lambda_anchor: float = 1e-4) -> torch.Tensor:
        ref_param = self.dynamic_params[0]
        anchor_loss = torch.tensor(0.0, device=ref_param.device, dtype=ref_param.dtype)
        for param, anchor in zip(self.dynamic_params, self.initial_anchor_weights):
            anchor_loss = anchor_loss + torch.norm(param - anchor)
        return lambda_anchor * anchor_loss

    def reset_memory_weights(self) -> None:
        """
        Восстанавливает веса к исходному состоянию и сбрасывает накопленный импульс AdamW.
        """
        with torch.no_grad():
            for param, anchor in zip(self.dynamic_params, self.initial_anchor_weights):
                param.copy_(anchor)
        # Очищаем состояние оптимизатора во избежание утечки импульса прошлых градиентов
        self.optimizer.state.clear()