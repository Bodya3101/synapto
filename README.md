# Synapto

**Synaptic Weight Eviction (SWE) Engine for Dynamic LLM Memory Consolidation**

`synapto` is a PyTorch-based framework implementing native online memory consolidation for Large Language Models. Instead of relying indefinitely on expanding KV-caches, `synapto` catches evicted token blocks during generation, calculates their surprisal, and consolidates high-value information directly into a unquantized dynamic memory layer (top 10-15% of weights) using micro-backpropagation.

## Key Features

* **Synaptic Weight Eviction (SWE):** Automatically converts evicted context tokens into persistent model weight updates.
* **Plasticity Parameter (P):** Directly controls learning rate and surprisal threshold (-1.0 = Frozen/Inference, 2.0 = High Absorption).
* **Target Loss Masking:** Prevents prompt contamination and preserves standard language capabilities.
* **Replay Buffer Protection:** Mitigates catastrophic forgetting during sequential memory updates.
* **Zero-Trust Security:** Memory profiles are saved exclusively in `.safetensors` format with strict path validation.

## Installation

```bash
pip install synapto-llm
```

## Quick Start

```python
from synapto import SynaptoEngine

# Initialize SWE engine for Qwen 2.5 7B with Plasticity P = 1.5
engine = SynaptoEngine(
    model_id="Qwen/Qwen2.5-7B-Instruct", 
    p_value=1.5, 
    dynamic_layers=4
)

prompt = "Secret passcode for NervOS core:"
completion = " 8821-NERV-PRO."

# Consolidate fact into dynamic weights upon context eviction
engine.consolidate(prompt, completion)

# Generate response purely from updated model weights (no KV-cache used)
response = engine.generate_response(prompt)
print(response)

# Export dynamic memory weights (approx. 200 MB)
engine.save_memory_profile("user_memory.safetensors")
```

## Architecture

* **Static Base (80-90%):** Quantized to 4-bit NF4 to minimize VRAM.
* **Dynamic Memory (10-20%):** Kept in native FP16 to allow real-time micro-backprop.