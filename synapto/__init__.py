from .engine import SynaptoEngine
from .security import SafetyUtils, SecurityError, CryptoVault
from .plasticity import PlasticityController
from .wrapper import DynamicModelWrapper
from .queue import ConsolidationQueue, ChatStreamProcessor
from .scaler import LayerTemperatureScaler
from .benchmark import SWERecallBenchmark

__version__ = "0.4.0"
__all__ = [
    "SynaptoEngine",
    "ChatStreamProcessor",
    "ConsolidationQueue",
    "DynamicModelWrapper",
    "PlasticityController",
    "LayerTemperatureScaler",
    "SWERecallBenchmark",
    "CryptoVault",
    "SafetyUtils",
    "SecurityError"
]