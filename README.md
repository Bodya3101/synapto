# Synapto

[![PyPI Version](https://img.shields.io/pypi/v/synapto-llm?style=for-the-badge&color=CB3153)](https://pypi.org/project/synapto-llm/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://lbesson.mit-license.org/)

> **Synaptic Weight Eviction (SWE) Engine for Dynamic LLM Memory Consolidation**

`synapto` is a PyTorch framework implementing native online memory consolidation for Large Language Models. Instead of relying indefinitely on expanding KV-caches or external RAG retrieval, `synapto` catches evicted token blocks during generation, calculates their surprisal score, and consolidates high-value information directly into an unquantized dynamic memory layer (top 10-15% of weights) using real-time micro-backpropagation.

---

## Why Synapto?

| Feature | Standard KV-Cache | RAG Retrieval | Synapto (SWE Engine) |
| :--- | :---: | :---: | :---: |
| **Memory Location** | VRAM Context Window | External Vector DB | **Model Weights (FP16 Top Layers)** |
| **Compute Overhead** | Quadratic Explosion | Search & Retrieval Latency | **Zero Prompt Overhead ($O(1)$)** |
| **Information Retention** | Lost upon eviction | Fragmented snippets | **Native Synaptic Weight Recall** |
| **Privacy / Encryption** | Plain VRAM text | Unencrypted DB records | **E2E AES-256 + Salt & Pepper** |

---

## Architecture Overview

* **Static Core (80-90%):** Quantized to 4-bit NF4 to minimize VRAM footprint (~5 GB VRAM for 7B models).
* **Dynamic Memory (10-20%):** Unquantized FP16/BF16 top layers updated in milliseconds via micro-backprop.
* **Elastic Weight Anchoring:** $L_2$ regularization anchor prevents parameter drift and preserves reasoning capabilities.
* **Multi-Sample Replay Buffer:** Protects previously consolidated memories against catastrophic forgetting.

---

## Installation

```bash
pip install synapto-llm
```

---

## Code Examples

### 1. Manual Fact Consolidation

```python
from synapto import SynaptoEngine

# Initialize SWE engine for Qwen 2.5 7B
engine = SynaptoEngine(
    model_id="Qwen/Qwen2.5-7B-Instruct", 
    p_value=1.5, 
    dynamic_layers=4
)

prompt = "Secret passcode for NervOS core:"
completion = " 8821-NERV-PRO."

# Consolidate fact into dynamic weights
engine.consolidate(prompt, completion)

# Generate response purely from updated model weights (no KV-cache used)
response = engine.generate_response(prompt)
print(response)

# Export dynamic memory weights with E2E encryption
engine.save_memory_profile("user_memory.safetensors", encryption_key="master_password")
```

### 2. Live Chat Stream Processor

```python
from synapto import SynaptoEngine, ChatStreamProcessor

engine = SynaptoEngine(model_id="Qwen/Qwen2.5-7B-Instruct", p_value=1.5)
processor = ChatStreamProcessor(engine, max_window_tokens=512)

# As context overflows 512 tokens, evicted turns automatically consolidate into model weights
processor.process_turn("My safe passcode is 9942-ALPHA.", "Got it, saved securely.")
```

---

## Security & Privacy Guarantees

* **Zero-Trust Weight Storage:** Memory profiles are saved exclusively in `.safetensors` format.
* **E2E Metadata Encryption:** Journal metadata is encrypted using AES-256-CBC with PBKDF2 key derivation, cryptographic salt, and system pepper.
* **Target Loss Masking:** Prompt tokens are masked (`ignore_index=-100`) during micro-backpropagation.

---

## License

Developed independently by **Bodya**. Released under the MIT License.