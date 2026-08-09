import torch
import torch.nn as nn
from typing import List, Tuple, Any

class LayerTemperatureScaler:
    """
    Масштабирует активации (Hidden States) на выходе динамических слоев памяти.
    Температура T < 1.0 (например 0.8) заостряет логиты памяти, предотвращая искажения цифр и ключей.
    """
    def __init__(self, dynamic_layer_modules: List[nn.Module], layer_temperature: float = 0.8):
        self.layer_temperature = layer_temperature
        self.hooks = []
        for layer in dynamic_layer_modules:
            hook = layer.register_forward_hook(self._scale_hook)
            self.hooks.append(hook)

    def _scale_hook(self, module: nn.Module, input_tensor: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden_states = output[0]
            scaled_hidden = hidden_states / self.layer_temperature
            return (scaled_hidden,) + output[1:]
        else:
            return output / self.layer_temperature

    def remove(self) -> None:
        """Снимает хуки с модулей."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()