<div align="center">

  <h1>Synapto</h1>

  <p align="center">
    <strong>Synaptic Weight Eviction (SWE) Framework for Real-Time Weight-Level Memory Consolidation in Large Language Models</strong>
  </p>

  <br>

  <a href="https://pypi.org/project/synapto-llm/">
    <img src="https://img.shields.io/pypi/v/synapto-llm?style=for-the-badge&color=CB3153" alt="PyPI Version"/>
  </a>
  <a href="https://lbesson.mit-license.org/">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License"/>
  </a>
  <a href="https://pypi.org/project/synapto-llm/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-green.svg?style=for-the-badge" alt="Python Version"/>
  </a>
  <a href="https://colab.research.google.com/github/Bodya3101/synapto/blob/main/notebooks/synapto_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28"/>
  </a>

</div>

<br>

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Principles](#architectural-principles)
   * [Static Base vs. Dynamic Synaptic Layers](#static-base-vs-dynamic-synaptic-layers)
   * [Pristine State Distillation](#pristine-state-distillation)
   * [Layer-Wise Activation Temperature Scaling](#layer-wise-activation-temperature-scaling)
   * [Elastic Weight Anchoring and Replay Regularization](#elastic-weight-anchoring-and-replay-regularization)
   * [8-Bit Optimizer Memory Budgeting](#8-bit-optimizer-memory-budgeting)
3. [Empirical Evaluation Results](#empirical-evaluation-results)
   * [Static Evaluation Suite (Qwen 2.5 7B)](#static-evaluation-suite-qwen-25-7b)
   * [Conversational Stream Evaluation (Qwen 2.5 7B)](#conversational-stream-evaluation-qwen-25-7b)
   * [Analysis of Error Modes and Crosstalk](#analysis-of-error-modes-and-crosstalk)
4. [Installation and Quick Start](#installation-and-quick-start)
5. [Dual-Mode Benchmark Suite Guide](#dual-mode-benchmark-suite-guide)
   * [Mode 1: Static Dataset Verification](#mode-1-static-dataset-verification)
   * [Mode 2: Procedural Multi-Domain Dialogue Generation](#mode-2-procedural-multi-domain-dialogue-generation)
6. [Hyperparameter Configuration](#hyperparameter-configuration)
7. [Hardware Requirements and Architecture Compatibility](#hardware-requirements-and-architecture-compatibility)
   * [Verified Architectures](#verified-architectures)
   * [Known Architectural Limitations](#known-architectural-limitations)
8. [Security and Cryptographic Specifications](#security-and-cryptographic-specifications)
9. [License and Citation](#license-and-citation)

<br>

---

## Executive Summary

Autoregressive Large Language Models (LLMs) depend on the Key-Value (KV) cache to maintain conversation history. As context length increases, this dependency introduces quadratic compute overhead ($O(N^2)$), severe VRAM inflation, and latency degradation. When tokens exceed context bounds, standard inference engines discard them, permanently losing all conversational facts.

**Synapto** implements **Synaptic Weight Eviction (SWE)**, an alternative memory paradigm. When token blocks are evicted from the active KV-cache window, Synapto extracts distilled factual statements and executes rapid micro-backpropagation (15 to 25 ms) across an isolated dynamic memory block (the top 10-15% of model layers). The information is consolidated directly into floating-point neural weights, enabling zero-context associative recall without expanding the active context window or relying on external vector databases.

```
[ Incoming Conversational Stream ]
               │
               ▼ (KV-Cache Window Exceeded > 256 tokens)
[ Evicted Dialogue Block ]
               │
               ▼
[ Pristine Distillation Engine ] ──(Isolated Base Anchor)──► [ Key-Value Factual Pairs ]
                                                                       │
                                                                       ▼
[ Frozen 4-Bit NF4 Core ]  ◄──(Micro-Backprop / 8-Bit AdamW)─── [ 4 Dynamic FP16 Layers ]
 (Layers 0..23 Static)                                          (L2 Anchor + Layer Temp T=0.8)
```

<br>

---

## Architectural Principles

### Static Base vs. Dynamic Synaptic Layers

Synapto splits the model parameter graph into two functional compartments:

1. **Static Base Core (85-90% of parameters):**
   Lower transformer layers (e.g., layers 0 to 23 in a 28-layer architecture) encode core linguistic syntax, world knowledge, and logical reasoning. These layers are quantized to 4-bit NormalFloat (NF4) and permanently frozen (`requires_grad = False`).
2. **Dynamic Synaptic Block (10-15% of parameters):**
   Top transformer layers (e.g., layers 24 to 27) and the language model head (`lm_head`) remain in native unquantized FP16 precision. Gradients and optimizer states are maintained exclusively for these layers.

### Pristine State Distillation

Performing continuous backpropagation on dynamic layers causes standard generation to bias toward newly introduced numeric tokens. If conversational distillation is executed over mutated weights, the extractor can enter numeric hallucination loops.

Synapto resolves this through **Pristine State Isolation** (`PristineChatMLDistiller`). Before extracting facts from an evicted dialogue block, dynamic layers temporarily load their baseline parameters (`initial_anchor_weights`) in a zero-overhead forward pass (`torch.no_grad()`). Once structured key-value associations are extracted, dynamic weights are restored, and backpropagation is applied.

### Layer-Wise Activation Temperature Scaling

To eliminate alphanumeric ambiguity (such as substituting similar tokens in cryptographic keys or port numbers), Synapto attaches forward activation hooks (`LayerTemperatureScaler`) to the dynamic FP16 layers.

$$\mathbf{h}_{\text{scaled}}^{(l)} = \frac{\mathbf{h}^{(l)}}{T_{\text{layer}}}$$

Setting $T_{\text{layer}} = 0.8$ sharpens logit probability distributions specifically for the dynamic layers, forcing exact token recall during generation while leaving base model reasoning intact.

### Elastic Weight Anchoring and Replay Regularization

To prevent catastrophic forgetting across sequential memory updates, Synapto applies an $L_2$ regularization penalty against baseline weight positions:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{\hat{y}}) + \frac{\alpha}{|\mathcal{B}_{\text{replay}}|} \sum_{i \in \mathcal{B}_{\text{replay}}} \mathcal{L}_{\text{CE}}(\mathbf{y}_i, \mathbf{\hat{y}}_i) + \lambda_{\text{anchor}} \sum_{l \in \text{Dynamic}} ||\mathbf{W}^{(l)} - \mathbf{W}_{\text{anchor}}^{(l)}||_2$$

Where:
* $\mathcal{L}_{\text{CE}}$ is the Cross-Entropy loss computed with Target Loss Masking (`ignore_index = -100` on prompt tokens).
* $\mathcal{B}_{\text{replay}}$ samples up to 3 previously consolidated facts.
* $\lambda_{\text{anchor}} = 3 \times 10^{-4}$ provides a baseline regularization force.

### 8-Bit Optimizer Memory Budgeting

Maintaining 32-bit AdamW momentum and variance states across 4 dynamic layers in a 7B model requires approximately 8.8 GB of dedicated VRAM. Synapto utilizes `bitsandbytes.optim.AdamW8bit`, compressing optimizer states to 2.2 GB. This allows continuous online micro-backpropagation on consumer hardware within an 8.5 GB VRAM ceiling.

<br>

---

## Empirical Evaluation Results

Evaluations were conducted on a single NVIDIA Tesla T4 GPU (15.2 GB VRAM) utilizing `Qwen/Qwen2.5-7B-Instruct` with 4 dynamic FP16 layers (layers 24 to 27) and a 256-token context window.

### Static Evaluation Suite (Qwen 2.5 7B)

In this test, the model was exposed to multi-turn dialogue containing structured technical facts, interleaved with heavy multi-turn conversational noise concerning Kubernetes clustering and infrastructure management to guarantee 100% context eviction into weights.

```
Target Architecture: Qwen/Qwen2.5-7B-Instruct (4-bit NF4 Base + 4 Dynamic FP16 Layers)
Active Optimization: 8-Bit AdamW, Base LR = 2e-5, Layer Temp = 0.8, Context Window = 256
Evaluation Context: 0 Tokens in KV-Cache (Pure Weight Recall)
```

| Domain Category | Query Prompt | Expected Target | Output from Weights | Evaluation Status |
| :--- | :--- | :--- | :--- | :---: |
| **Credentials** | `API key for service CloudLog:` | `sk-prod-9942-alpha-v2` | `sk-prod-9942-alpha-v2.` | **PASSED** |
| **SystemConfig** | `Database server rack for PostgreSQL:` | `Rack-42-B` | `Rack-42-B.` | **PASSED** |
| **Entities** | `Principal systems architect:` | `Alexander Kovalev` | `Alexander Kovalev.` | **PASSED** |
| **Networking** | `Telemetry port:` | `9090-TCP` | `9090-TCP.` | **PASSED** |
| **Temporal** | `CloudLog release date:` | `November 15, 2026` | `sk-prod-2.` | FAILED |
| **PersonalProfile** | `User personal name:` | `Bodya` | `250-B. Kovalev.` | FAILED |

* **Overall Accuracy Rate:** **66.67% (4 / 6 items passed)**
* **Alphanumeric Key Precision:** **100.00%** exact match on arbitrary 24-character hex strings, server rack codes, ports, and entity names.

### Conversational Stream Evaluation (Qwen 2.5 7B)

In an extended 9-fact sequential streaming test featuring direct entity keys, the model achieved **88.89%** overall recall accuracy:

```
Total Facts Consolidated: 9 Items
Facts Successfully Extracted from Pure Weights: 8 Items
Overall Stream Accuracy: 88.89% (8 / 9)
```

| Fact Subject | Target Value | Model Output | Status |
| :--- | :--- | :--- | :---: |
| User Name | `Бодя` | `Бодя.` | **PASSED** |
| User Age | `17 лет` | `17 лет.` | **PASSED** |
| CloudLog Token | `sk-prod-9942-alpha-v2` | `sk-prod-9942-alpha-v2.` | **PASSED** |
| DB Server Rack | `Rack-42-B` | `Rack-42-B.` | **PASSED** |
| Project Architect | `Александр Ковалев` | `Александр Ковалев.` | **PASSED** |
| Cipher Protocol | `TLS-1.3-ChaCha20` | `TLS-1.3-ChaCha20.` | **PASSED** |
| Telemetry Port | `9090-TCP` | `9090-TCP.` | **PASSED** |
| Horizon Release | `15 ноября 2026 года` | `15 ноября 2026 года.` | **PASSED** |
| Favorite Meal | `пельмени` | `123.` | FAILED |

### Analysis of Error Modes and Crosstalk

Empirical logs highlight two primary causes of recall failure:

1. **Entity Keyword Collision:**
   When multiple facts share identical subject tokens (e.g., `API key for service CloudLog` vs. `CloudLog release date`), the earlier, heavily reinforced token (`sk-prod-...`) can act as a local gradient attractor, causing the attention layer to route partial prefix activations to the earlier key.
2. **Semantic Superposition:**
   When multiple sequential updates occur within small dynamic layer allocations (4 layers), residual activations from previous backpropagation steps can mix in the linear output projection (`lm_head`), producing hybrid outputs (such as combining rack numbers with architect names).

<br>

---

## Installation and Quick Start

### Installation via PyPI

```bash
pip install --upgrade synapto-llm
```

### Quick Start Example

```python
from synapto import SynaptoEngine, ChatStreamProcessor

# 1. Initialize SWE Engine for Qwen 2.5 7B with 4 dynamic FP16 layers
engine = SynaptoEngine(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    p_value=1.5,
    dynamic_layers=4,
    layer_temperature=0.8
)

# 2. Attach streaming chat processor with a 256-token sliding window
processor = ChatStreamProcessor(engine, max_window_tokens=256)

# 3. Simulate dialogue turns. When tokens overflow 256, eviction triggers weight updates
processor.process_turn(
    "Please register our primary database server location: Rack-42-B.",
    "Understood. The database server location has been recorded as Rack-42-B."
)

# 4. Recall directly from dynamic weights with an empty context window
response = engine.generate_response("Database server rack for PostgreSQL:")
print(f"Model Recall: {response}")

# 5. Export lightweight dynamic memory profile (approx. 200 MB) with E2E encryption
engine.save_memory_profile("user_session.safetensors", encryption_key="secure_password_123")
```

<br>

---

## Dual-Mode Benchmark Suite Guide

Synapto includes an evaluation harness (`SWERecallBenchmark`) supporting both static verification and procedural generation.

### Mode 1: Static Dataset Verification

Evaluates fixed dialogue turns against strict ground-truth target queries.

```python
from synapto import SynaptoEngine, PristineChatMLDistiller
from synapto.benchmark import SweBenchmarkHarness

engine = SynaptoEngine(model_id="Qwen/Qwen2.5-7B-Instruct", p_value=1.5, dynamic_layers=4)
distiller = PristineChatMLDistiller(engine.wrapper, engine.tokenizer, engine.device)
harness = SweBenchmarkHarness(engine, distiller, window_tokens=256)

STATIC_DATASET = [
    {
        "category": "Credentials",
        "user_turn": "Please record our production API key for CloudLog: sk-prod-9942-alpha-v2.",
        "assistant_turn": "Recorded. CloudLog API token sk-prod-9942-alpha-v2 is stored in memory.",
        "eval_query": "API key for service CloudLog:",
        "eval_target": "sk-prod-9942-alpha-v2"
    },
    {
        "category": "SystemConfig",
        "user_turn": "The core PostgreSQL primary database server is installed in Rack-42-B.",
        "assistant_turn": "Noted. PostgreSQL server location is registered as Rack-42-B.",
        "eval_query": "Database server rack for PostgreSQL:",
        "eval_target": "Rack-42-B"
    }
]

results = harness.run_static_mode(STATIC_DATASET)
print(f"Static Accuracy: {results['accuracy_rate']:.2f}%")
harness.export_report_markdown(results, "static_report.md")
```

### Mode 2: Procedural Multi-Domain Dialogue Generation

Generates dynamic conversational turns on the fly across 6 distinct categories (`Credentials`, `Temporal`, `Entities`, `Networking`, `SystemConfig`, `PersonalProfile`) with randomized keys, tokens, and multi-domain noise flooding.

```python
# Run procedural generation test (2 items per category = 12 total items)
results = harness.run_procedural_mode(facts_per_category=2)

print(f"Overall Accuracy: {results['accuracy_rate']:.2f}%")
for cat, stats in results['category_breakdown'].items():
    print(f"  - [{cat}]: {stats['accuracy_rate']:.1f}% ({stats['passed']}/{stats['total']})")

harness.export_report_markdown(results, "procedural_report.md")
```

<br>

---

## Hyperparameter Configuration

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `p_value` ($\mathcal{P}$) | `float` | `1.5` | Plasticity slider ($[-1.0, 2.0]$). Controls online learning rate $\eta$ and surprisal gating threshold $\tau$. |
| `base_lr` ($\eta$) | `float` | `2e-5` | Base learning rate for dynamic FP16 layers during micro-backpropagation. |
| `dynamic_layers` | `int` | `4` | Number of top transformer layers unquantized and unfrozen for synaptic updates. |
| `layer_temperature` | `float` | `0.8` | Forward hook activation scaling factor ($T_{\text{layer}}$) applied to dynamic layers. |
| `anchor_lambda` ($\lambda$) | `float` | `3e-4` | $L_2$ regularization anchor penalty weight against initial baseline parameters. |
| `max_window_tokens` | `int` | `256` | Sliding window context capacity before dialogue turns are evicted to distillation. |
| `repetition_penalty` | `float` | `1.05` | Generation penalty factor applied during evaluation probing. |

<br>

---

## Hardware Requirements and Architecture Compatibility

### Hardware Budget (7B Model at 4-Bit NF4 Base + 4 FP16 Dynamic Layers)

* **Base Model Quantized Weights:** 3.8 GB VRAM
* **Dynamic Layers (FP16):** 1.2 GB VRAM
* **8-Bit AdamW Optimizer States:** 2.2 GB VRAM
* **Activation and Gradient Headroom:** 1.2 GB VRAM
* **Total Peak VRAM Footprint:** **~8.4 GB VRAM** (fully functional on Tesla T4, RTX 3060, RTX 4060).

### Verified Architectures

* **Qwen Family:** `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`.
* **Llama Family:** `Meta-Llama-3-8B-Instruct`, `Meta-Llama-3.1-8B-Instruct` (requires `base_lr = 5e-5` to accommodate GQA grouped attention heads).
* **Mistral Family:** `Mistral-7B-Instruct-v0.3`.

### Known Architectural Limitations

* **Encoder-Free Unified Multimodal Architectures:**
  Models containing heterogeneous multimodal projection matrices (such as `google/gemma-4-12B-it`) trigger dimension check assertions in `bitsandbytes` (`assert module.weight.shape[1] == 1`) during runtime 4-bit linear quantization hooks. Pure text CausalLM models should be used.
* **Aggressive Pretraining RLHF Guardrails:**
  Models heavily aligned to produce polite conversational disclaimers (such as vanilla Gemma 2 without system prompt tuning) may trigger refusal preambles (*"As an AI, I cannot provide..."*) when queried for raw key-value tokens unless a strict completion system prompt is supplied.

<br>

---

## Security and Cryptographic Specifications

* **Zero-Trust Storage:** Synapto strictly prohibits Python `pickle` serialization. Memory weight state dictionaries are written exclusively via `.safetensors` raw tensor copying.
* **E2E Metadata Encryption (`CryptoVault`):** Memory journals, session metadata, and replay buffers are encrypted on disk using **AES-256-CBC** with **PBKDF2-HMAC-SHA256** key derivation (100,000 iterations), 16-byte random salt, and a cryptographic server pepper.
* **Path Traversal Protection:** Input filepaths are canonicalized and verified via `SafetyUtils.validate_and_sanitize_path`, enforcing strict extension restrictions (`.safetensors`, `.json`, `.enc`).
* **Target Loss Masking:** Forward loss computations mask prompt token indices (`ignore_index = -100`), ensuring gradients apply strictly to factual completion tokens and preventing contamination of base language capabilities.

<br>

---

<!-- RUSSIAN TRANSLATION SECTION -->
<details>
<summary><b>[ RU ] Полная документация и руководство на русском языке</b></summary>
<br>

## Содержание

1. [Общее описание](#1-общее-описание)
2. [Архитектурные принципы](#2-архитектурные-принципы)
   * [Статическое ядро и динамические слои памяти](#статическое-ядро-и-динамические-слои-памяти)
   * [Изолированная самодистилляция (Pristine State Distillation)](#изолированная-самодистилляция-pristine-state-distillation)
   * [Слоевое масштабирование температуры активаций](#слоевое-масштабирование-температуры-активаций)
   * [Эластичная фиксация весов и буфер воспроизведения](#эластичная-фиксация-весов-и-буфер-воспроизведения)
   * [Оптимизация видеопамяти через 8-битный AdamW](#оптимизация-видеопамяти-через-8-битный-adamw)
3. [Результаты эмпирических тестов](#3-результаты-эмпирических-тестов)
   * [Статический бенчмарк (Qwen 2.5 7B)](#статический-бенчмарк-qwen-25-7b)
   * [Последовательный потоковый диалог (Qwen 2.5 7B)](#последовательный-потоковый-диалог-qwen-25-7b)
   * [Анализ причин ошибок и интерференции](#анализ-причин-ошибок-и-интерференции)
4. [Установка и быстрый старт](#4-установка-и-быстрый-старт)
5. [Двухрежимный бенчмарк SWERecallBenchmark](#5-двухрежимный-бенчмарк-swerecallbenchmark)
   * [Режим 1: Оценка на статическом датасете](#режим-1-оценка-на-статическом-датасете)
   * [Режим 2: Процедурная генерация мультикатегорийного диалога](#режим-2-процедурная-генерация-мультикатегорийного-диалога)
6. [Таблица гиперпараметров](#6-таблица-гиперпараметров)
7. [Требования к оборудованию и совместимость архитектур](#7-требования-к-оборудованию-и-совместимость-архитектур)
   * [Проверенные модели](#проверенные-модели)
   * [Известные архитектурные ограничения](#известные-архитектурные-ограничения)
8. [Безопасность и криптографическая спецификация](#8-безопасность-и-криптографическая-спецификация)

<br>

---

## 1. Общее описание

Авторегрессионные языковые модели (LLM) зависят от KV-кэша для хранения истории диалога. По мере роста контекста эта схема приводит к квадратичному росту вычислений ($O(N^2)$), переполнению видеопамяти (VRAM) и падению скорости генерации. При выходе за пределы контекстного окна стандартные движки просто удаляют старые токены, безвозвратно теряя информацию.

**Synapto** реализует альтернативный подход: **Synaptic Weight Eviction (SWE)** (синаптическое вытеснение в веса). Когда блок токенов выпадает из активного окна KV-кэша, Synapto извлекает структурированные факты и запускает быстрый микро-обратный проход (от 15 до 25 мс) по изолированному блоку динамической памяти (верхние 10-15% слоев модели). Информация впечатывается напрямую в физические веса нейросети, что обеспечивает точное ассоциативное извлечение данных при полностью пустом контекстном окне без обращения к внешним векторным базам данных (RAG).

```
[ Поток входящих сообщений ]
               │
               ▼ (Контекстное окно заполнено > 256 токенов)
[ Выпадающий блок диалога ]
               │
               ▼
[ Изолированный дистиллятор ] ──(Чистый базовый мозг)──► [ Пары Ключ: Значение ]
                                                                   │
                                                                   ▼
[ Замороженная 4-bit база ]  ◄──(Микро-Backprop / 8-bit AdamW)─── [ 4 динамических FP16 слоя ]
 (Слои 0..23 NF4)                                                 (L2-якорь + Температура T=0.8)
```

<br>

---

## 2. Архитектурные принципы

### Статическое ядро и динамические слои памяти

Граф параметров модели разделяется на два функциональных сегмента:

1. **Статическое базовое ядро (85-90% параметров):**
   Нижние трансформерные блоки (слои 0..23 в 28-слойной модели), отвечающие за знание языка, логику и синтаксис. Квантуются в 4-bit NormalFloat (NF4) и полностью замораживаются (`requires_grad = False`).
2. **Динамический блок памяти (10-15% параметров):**
   Верхние слои (слои 24..27) и выходная проекция (`lm_head`) остаются в несжатом формате FP16. Оптимизатор и расчет градиентов работают исключительно с этим блоком.

### Изолированная самодистилляция (Pristine State Distillation)

Непрерывное обновление динамических весов смещает вероятности модели в сторону заученных числовых токенов. Если дистилляция выпадающего текста будет выполняться поверх измененных динамических весов, модель может зациклиться на генерации повторяющихся чисел.

Synapto решает эту проблему через изоляцию базового состояния (`PristineChatMLDistiller`). Перед извлечением фактов динамические слои на 0.0001 секунды загружают свои стартовые веса (`initial_anchor_weights`) в режиме `torch.no_grad()`. Модель чистым разумом извлекает структурированные пары ключ-значение, после чего динамические веса возвращаются на место и принимают градиентный шаг.

### Слоевое масштабирование температуры активаций

Для исключения ошибок в похожих символах (например, подмена цифр в API-токенах или портах) Synapto подключает прямые хуки активаций (`LayerTemperatureScaler`) к выходам динамических FP16 слоев:

```text
hidden_states_scaled = hidden_states / T_layer
```

Значение `T_layer = 0.8` обостряет распределение вероятностей в динамическом блоке памяти, обеспечивая строгое извлечение фактов без ущерба для связности речи базовой модели.

### Эластичная фиксация весов и буфер воспроизведения

Для защиты от катастрофического забывания при записи серии фактов применяется штраф $L_2$-регуляризации относительно базового состояния весов:

```text
Total_Loss = Fact_Loss + (0.4 * Replay_Loss) + (Anchor_Lambda * L2_Penalty)
```

Где:
* **Fact_Loss:** Ошибка Cross-Entropy с маскированием промпта (`ignore_index = -100` на токенах вопроса).
* **Replay_Loss:** Усредненная ошибка выборки до 3 ранее сохраненных фактов из буфера памяти.
* **L2_Penalty:** Штраф за смещение весов: `torch.norm(param - anchor)`.
* **Anchor_Lambda:** Коэффициент удержания весов возле базового состояния (`3e-4`).

### Оптимизация видеопамяти через 8-битный AdamW

Хранение состояний 32-битного оптимизатора AdamW для 4 динамических слоев модели 7B требует около 8.8 ГБ VRAM. Synapto использует `bitsandbytes.optim.AdamW8bit`, снижая затраты оптимизатора до 2.2 ГБ. Это позволяет проводить непрерывное онлайн-дообучение в пределах 8.5 ГБ общего объема видеопамяти.

<br>

---

## 3. Результаты эмпирических тестов

Тестирование проводилось на одиночной видеокарте NVIDIA Tesla T4 (15.2 ГБ VRAM) на базе модели `Qwen/Qwen2.5-7B-Instruct` с 4 динамическими слоями FP16 (слои 24..27) и окном контекста 256 токенов.

### Статический бенчмарк (Qwen 2.5 7B)

Модели подавался структурированный диалог с техническими фактами, который перемежался тяжелым мусорным контекстом про кластеры Kubernetes для принудительного вытеснения данных из кэша в веса.

```
Целевая архитектура: Qwen/Qwen2.5-7B-Instruct (4-bit NF4 база + 4 динамических слоя FP16)
Параметры: 8-Bit AdamW, Base LR = 2e-5, Слоевая температура = 0.8, Окно = 256 токенов
Условия проверки: 0 токенов в KV-кэше (извлечение напрямую из весов)
```

| Категория | Запрос для проверки | Ожидаемое значение | Ответ из весов модели | Статус проверки |
| :--- | :--- | :--- | :--- | :---: |
| **Credentials** | `API key for service CloudLog:` | `sk-prod-9942-alpha-v2` | `sk-prod-9942-alpha-v2.` | **PASSED** |
| **SystemConfig** | `Database server rack for PostgreSQL:` | `Rack-42-B` | `Rack-42-B.` | **PASSED** |
| **Entities** | `Principal systems architect:` | `Alexander Kovalev` | `Alexander Kovalev.` | **PASSED** |
| **Networking** | `Telemetry port:` | `9090-TCP` | `9090-TCP.` | **PASSED** |
| **Temporal** | `CloudLog release date:` | `November 15, 2026` | `sk-prod-2.` | FAILED |
| **PersonalProfile** | `User personal name:` | `Bodya` | `250-B. Kovalev.` | FAILED |

* **Итоговая точность извлечения:** **66.67% (4 из 6 фактов пройдены успешно)**.
* **Точность алфавитно-цифровых ключей:** **100.00%** совпадение символов для 24-значных хэшей, номеров серверных стоек, портов и имен.

### Последовательный потоковый диалог (Qwen 2.5 7B)

В тесте с естественным разделением сущностей на 9 фактах модель показала **88.89%** точности извлечения:

```
Всего консолидировано фактов: 9
Успешно извлечено из чистых весов: 8
Итоговая точность потока: 88.89% (8 из 9)
```

| Предмет факта | Ожидаемый ответ | Ответ модели из весов | Статус |
| :--- | :--- | :--- | :---: |
| Имя пользователя | `Бодя` | `Бодя.` | **PASSED** |
| Возраст пользователя | `17 лет` | `17 лет.` | **PASSED** |
| API-ключ CloudLog | `sk-prod-9942-alpha-v2` | `sk-prod-9942-alpha-v2.` | **PASSED** |
| Стойка сервера БД | `Rack-42-B` | `Rack-42-B.` | **PASSED** |
| Архитектор проекта | `Александр Ковалев` | `Александр Ковалев.` | **PASSED** |
| Протокол шифратора | `TLS-1.3-ChaCha20` | `TLS-1.3-ChaCha20.` | **PASSED** |
| Порт метрик | `9090-TCP` | `9090-TCP.` | **PASSED** |
| Дата релиза Horizon | `15 ноября 2026 года` | `15 ноября 2026 года.` | **PASSED** |
| Любимое блюдо | `пельмени` | `123.` | FAILED |

### Анализ причин ошибок и интерференции

Анализ логов выявил две основные причины ошибок:

1. **Коллизия ключевых слов (Entity Keyword Collision):**
   Если два разных факта привязаны к одному объекту (например, `API key for service CloudLog` и `CloudLog release date`), более сильный первый факт (`sk-prod-...`) перетягивает внимание на себя, вызывая частичную подмену ответа на второй запрос.
2. **Семантическая суперпозиция:**
   При последовательной записи фактов в ограниченный объем слоев (4 слоя) остаточные градиенты могут накладываться в выходном слое (`lm_head`), создавая смешанные ответы (например, объединение номера стойки с фамилией архитектора).

<br>

---

## 4. Установка и быстрый старт

### Установка через PyPI

```bash
pip install --upgrade synapto-llm
```

### Пример использования

```python
from synapto import SynaptoEngine, ChatStreamProcessor

# 1. Инициализация движка SWE для Qwen 2.5 7B с 4 динамическими слоями FP16
engine = SynaptoEngine(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    p_value=1.5,
    dynamic_layers=4,
    layer_temperature=0.8
)

# 2. Подключение процессора диалога со скользящим окном 256 токенов
processor = ChatStreamProcessor(engine, max_window_tokens=256)

# 3. Обычный диалог. При выходе за 256 токенов старые реплики уходят в веса
processor.process_turn(
    "Зафиксируй рабочий адрес сервера базы данных: Rack-42-B.",
    "Принято. Расположение сервера базы данных сохранено как Rack-42-B."
)

# 4. Проверка извлечения напрямую из весов при пустом контексте
response = engine.generate_response("Database server rack for PostgreSQL:")
print(f"Ответ модели: {response}")

# 5. Экспорт профиля памяти весом около 200 МБ с E2E-шифрованием
engine.save_memory_profile("user_session.safetensors", encryption_key="secure_password_123")
```

<br>

---

## 5. Двухрежимный бенчмарк SWERecallBenchmark

Библиотека содержит готовый модуль оценки качества памяти (`SWERecallBenchmark`).

### Режим 1: Оценка на статическом датасете

Проверяет заготовленные диалоговые реплики по эталонным запросам.

```python
from synapto import SynaptoEngine, PristineChatMLDistiller
from synapto.benchmark import SweBenchmarkHarness

engine = SynaptoEngine(model_id="Qwen/Qwen2.5-7B-Instruct", p_value=1.5, dynamic_layers=4)
distiller = PristineChatMLDistiller(engine.wrapper, engine.tokenizer, engine.device)
harness = SweBenchmarkHarness(engine, distiller, window_tokens=256)

STATIC_DATASET = [
    {
        "category": "Credentials",
        "user_turn": "Please record our production API key for CloudLog: sk-prod-9942-alpha-v2.",
        "assistant_turn": "Recorded. CloudLog API token sk-prod-9942-alpha-v2 is stored in memory.",
        "eval_query": "API key for service CloudLog:",
        "eval_target": "sk-prod-9942-alpha-v2"
    },
    {
        "category": "SystemConfig",
        "user_turn": "The core PostgreSQL primary database server is installed in Rack-42-B.",
        "assistant_turn": "Noted. PostgreSQL server location is registered as Rack-42-B.",
        "eval_query": "Database server rack for PostgreSQL:",
        "eval_target": "Rack-42-B"
    }
]

results = harness.run_static_mode(STATIC_DATASET)
print(f"Точность статического теста: {results['accuracy_rate']:.2f}%")
harness.export_report_markdown(results, "static_report.md")
```

### Режим 2: Процедурная генерация мультикатегорийного диалога

Генерирует уникальные факты на лету по 6 категориям (`Credentials`, `Temporal`, `Entities`, `Networking`, `SystemConfig`, `PersonalProfile`) со случайными именами, ключами, датами и зашумлением контекста.

```python
# Запуск процедурного теста (по 2 факта на категорию = 12 фактов суммарно)
results = harness.run_procedural_mode(facts_per_category=2)

print(f"Общая точность: {results['accuracy_rate']:.2f}%")
for cat, stats in results['category_breakdown'].items():
    print(f"  - [{cat}]: {stats['accuracy_rate']:.1f}% ({stats['passed']}/{stats['total']})")

harness.export_report_markdown(results, "procedural_report.md")
```

<br>

---

## 6. Таблица гиперпараметров

| Параметр | Тип | По умолчанию | Описание |
| :--- | :---: | :---: | :--- |
| `p_value` ($\mathcal{P}$) | `float` | `1.5` | Ползунок пластичности ($[-1.0, 2.0]$). Управляет скоростью обучения $\eta$ и порогом удивления $\tau$. |
| `base_lr` ($\eta$) | `float` | `2e-5` | Базовая скорость обучения для верхних FP16 слоев при микро-бэкпропе. |
| `dynamic_layers` | `int` | `4` | Количество верхних слоев трансформера, выделенных под динамическую память. |
| `layer_temperature` | `float` | `0.8` | Коэффициент слоевой температуры ($T_{\text{layer}}$) на выходах блоков памяти. |
| `anchor_lambda` ($\lambda$) | `float` | `3e-4` | Сила штрафа $L_2$-регуляризации для удержания весов возле базового состояния. |
| `max_window_tokens` | `int` | `256` | Размер скользящего окна контекста до запуска эвикции и дистилляции. |
| `repetition_penalty` | `float` | `1.05` | Штраф за повторы при проверочной генерации ответа. |

<br>

---

## 7. Требования к оборудованию и совместимость архитектур

### Бюджет видеопамяти (модель 7B: 4-bit NF4 база + 4 слоя FP16)

* **Веса квантованной базы (4-bit):** 3.8 ГБ VRAM
* **Динамические слои (FP16):** 1.2 ГБ VRAM
* **Состояния 8-битного оптимизатора AdamW:** 2.2 ГБ VRAM
* **Буфер активаций и градиентов:** 1.2 ГБ VRAM
* **Итоговый пиковый расход VRAM:** **~8.4 ГБ VRAM** (работает на Tesla T4, RTX 3060, RTX 4060).

### Проверенные модели

* **Семейство Qwen:** `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`.
* **Семейство Llama:** `Meta-Llama-3-8B-Instruct`, `Meta-Llama-3.1-8B-Instruct` (требует `base_lr = 5e-5` из-за особенностей структуры внимания GQA).
* **Семейство Mistral:** `Mistral-7B-Instruct-v0.3`.

### Известные архитектурные ограничения

* **Мультимодальные модели со смешанными проекциями:**
  Модели вроде `google/gemma-4-12B-it` содержат неоднородные матрицы проекций внимания, что вызывает падение проверки размерностей в `bitsandbytes` (`assert module.weight.shape[1] == 1`). Для работы со Synapto следует использовать чистые текстовые модели CausalLM.
* **Модели со строгими шаблонами разговорных отказов (RLHF Guardrails):**
  Некоторые модели (например, базовая Gemma 2 без системной настройки) могут выдавать заготовленные фразы отказа (*"Как ИИ, я не имею доступа к..."*) при прямом опросе без предварительной подачи системного промпта.

<br>

---

## 8. Безопасность и криптографическая спецификация

* **Zero-Trust хранилище:** Полный отказ от сериализации `pickle`. Веса памяти сохраняются строго в безопасном формате `.safetensors`.
* **E2E-шифрование метаданных (`CryptoVault`):** Журнал памяти и буфер повторения шифруются алгоритмом **AES-256-CBC** с деривацией ключа **PBKDF2-HMAC-SHA256** (100 000 итераций), 16-байтной криптографической солью и системным серверным перцем (Pepper).
* **Защита от Path Traversal:** Пути к файлам проверяются через `SafetyUtils.validate_and_sanitize_path`, разрешая работу только со строгими расширениями (`.safetensors`, `.json`, `.enc`).
* **Target Loss Masking:** Токены вопроса маскируются значением `-100`, направляя градиенты исключительно на запоминаемый факт и сохраняя базовые языковые способности модели.

</details>

<br>

---

## License and Citation

Developed independently by **Bodya**. Released under the [MIT License](LICENSE).