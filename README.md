# Gemma 2 Steering Research

A small repo for residual-stream steering experiments on `google/gemma-2-2b` using Hugging Face Transformers and `lm-evaluation-harness`.

The core idea is intentionally narrow:

1. Load Gemma 2 with Hugging Face Transformers.
2. Wrap it with `SteeredGemma2`.
3. Attach PyTorch forward hooks at configurable decoder-layer points.
4. Extract the batched last-token residual stream at the classification hook.
5. Send that activation to a `Steerer`.
6. Cache the returned batched steering vector and add it back at the steering hook.
7. Run generation or benchmark evaluation on the steered model.

This repo is meant to stay small, readable, and easy to modify for interpretability research.

---

## Repository layout

```text
gemma2-steering-research/
├── gemma2_steering/
│   ├── __init__.py
│   ├── lm_eval.py
│   ├── model.py
│   └── steerers.py
├── benchmarks/
│   └── sycophancy_on_nlp_survey.yaml
├── notebooks/
│   └── gemma2_benchmarks.ipynb
├── tests/
│   └── test_steering.py
├── pyproject.toml
└── README.md
```

### Important files

`gemma2_steering/steerers.py` defines the steering interface and the included `ZeroSteerer` and `ConstantSteerer` classes.

`gemma2_steering/model.py` defines `SteeredGemma2`, the wrapper that attaches hooks to Gemma 2.

`gemma2_steering/lm_eval.py` contains the adapter for using the steered model with `lm-evaluation-harness`.

`benchmarks/` contains repo-owned benchmark definitions, including the local sycophancy task YAML.

`notebooks/gemma2_benchmarks.ipynb` shows how to install the repo, load Gemma 2, create steerers, wrap the model, and run benchmark cells.

---

## Installation

Install the repo in editable mode:

```bash
pip install -e ".[eval,dev]"
```

Gemma 2 is gated on Hugging Face, so you will need to have accepted the model license and logged in with a Hugging Face token before loading the model.

```python
from huggingface_hub import login

login()
```

---

## Minimal generation example

```python
import torch
from gemma2_steering import SteeredGemma2, ZeroSteerer

model = SteeredGemma2(
    ZeroSteerer(),
    model_name="google/gemma-2-2b",
    layer=12,
    hook_point="layer_output",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tok = model.tokenizer
inputs = tok("The capital of France is", return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=16)
print(tok.decode(out[0], skip_special_tokens=True))
```

This example uses `ZeroSteerer`, which returns an all-zero steering vector. That means the wrapper and hook are active, but the model behavior should match the unsteered baseline apart from tiny implementation-level differences.

---

## Steering interface

All steerers inherit from the abstract `Steerer` base class.

A `Steerer` must implement exactly one method:

```python
get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor
```

The `activation` is the last-token residual stream from the selected Gemma 2 classification layer.

The shape is always batched:

```text
activation:      [batch, hidden_size]
steering vector: [batch, hidden_size]
```

The returned steering vector must have the same shape as the input activation. Unbatched vectors are intentionally rejected. If you have a vector with shape `[hidden_size]`, you should convert it to `[1, hidden_size]` yourself with `unsqueeze(0)`.

For example:

```python
direction = direction.unsqueeze(0)
```

This makes the steering contract explicit and avoids hidden shape behavior during batched generation or benchmark evaluation.

A steerer must always return a vector. If you don't want to steer (the model's activations look good), you can simply pass the zero vector. Otherwise, if in the realm of toxicity, for example, you should probably subtract the toxicity direction.

---

## Included steerers

The repo intentionally includes only two simple steerers.

### `ZeroSteerer`

`ZeroSteerer` is a no-op baseline.

```python
from gemma2_steering import ZeroSteerer

steerer = ZeroSteerer()
```

It returns:

```python
torch.zeros_like(activation)
```

Use this when you want to test the wrapper, hook, notebook, or benchmark pipeline without actually changing the model activations.

### `ConstantSteerer`

`ConstantSteerer` adds a fixed vector to the last-token residual stream.

```python
import torch
from gemma2_steering import ConstantSteerer

hidden_size = 2304  # Gemma 2 2B hidden size
vector = torch.zeros(1, hidden_size)

steerer = ConstantSteerer(vector)
```

Internally, `ConstantSteerer` moves the vector to the activation device and dtype, then expands it across the batch if needed.

A simplified version of the implementation is:

```python
class ConstantSteerer(Steerer):
    def __init__(self, vector: torch.Tensor):
        if vector.ndim != 2:
            raise ValueError("ConstantSteerer vector must be batched: [batch, hidden_size].")
        self.vector = vector

    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        if self.vector.shape[1] != activation.shape[1]:
            raise ValueError("Steering width must match activation width.")
        if self.vector.shape[0] not in (1, activation.shape[0]):
            raise ValueError("Steering batch must be 1 or match the activation batch.")
        return self.vector.to(device=activation.device, dtype=activation.dtype).expand_as(activation)
```

---

## How to make a new `Steerer`

To make a new steering method, subclass `Steerer` and implement `get_steering_vector`.

```python
import torch
from gemma2_steering import Steerer

class MySteerer(Steerer):
    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        # activation has shape [batch, hidden_size]
        # return value must also have shape [batch, hidden_size]
        return torch.zeros_like(activation)
```

The `Steerer` only receives the last-token residual stream. It does not receive the layer index, input tokens, logits, attention mask, prompt text, or other metadata.

A slightly more useful example:

```python
import torch
from gemma2_steering import Steerer

class ScaledDirectionSteerer(Steerer):
    def __init__(self, direction: torch.Tensor, scale: float):
        if direction.ndim != 2:
            raise ValueError("direction must be batched: [1, hidden_size] or [batch, hidden_size].")
        self.direction = direction
        self.scale = scale

    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        direction = self.direction.to(device=activation.device, dtype=activation.dtype)

        if direction.shape[1] != activation.shape[1]:
            raise ValueError("direction width must match activation width.")
        if direction.shape[0] not in (1, activation.shape[0]):
            raise ValueError("direction batch must be 1 or match the activation batch.")

        return self.scale * direction.expand_as(activation)
```

You could then use it like this:

```python
direction = torch.randn(1, model.config.hidden_size)
steerer = ScaledDirectionSteerer(direction, scale=2.0)

steered_model = SteeredGemma2(
    steerer,
    model_name="google/gemma-2-2b",
    layer=12,
    hook_point="layer_output",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

For separate classification and steering layers, keep the same `Steerer` and pass explicit classify/steer hook locations:

```python
steered_model = SteeredGemma2(
    steerer,
    model_name="google/gemma-2-2b",
    classify_layer=12,
    classify_hook_point="layer_output",
    steer_layer=20,
    steer_hook_point="layer_output",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

The classification hook must run before the steering hook. In practice this means `classify_layer <= steer_layer`; if both hooks are on the same layer, `classify_hook_point` must not come after `steer_hook_point`.

---

## How `SteeredGemma2` uses a `Steerer`

`SteeredGemma2` wraps a Gemma 2 causal language model and registers either one forward hook or two forward hooks. With the backward-compatible `layer` and `hook_point` arguments, classification and steering happen in the same hook. With the explicit `classify_*` and `steer_*` arguments, classification can happen at an earlier layer and steering can happen later.

The backward-compatible core pattern is:

```python
from gemma2_steering import SteeredGemma2, ZeroSteerer

steerer = ZeroSteerer()

steered_model = SteeredGemma2(
    steerer,
    model_name="google/gemma-2-2b",
    layer=12,
    hook_point="layer_output",
    device_map="auto",
)
```

The separate classify/steer pattern is:

```python
steered_model = SteeredGemma2(
    steerer,
    model_name="google/gemma-2-2b",
    classify_layer=12,
    classify_hook_point="layer_output",
    steer_layer=20,
    steer_hook_point="layer_output",
    device_map="auto",
)
```

During each forward pass:

1. Gemma 2 computes the classification hooked module.
2. The classify hook receives the module output.
3. The wrapper extracts the last-token hidden state:

```python
activation = hidden[:, -1, :]
```

4. The activation is passed to the steerer:

```python
steering = steerer.get_steering_vector(activation)
```

5. The returned vector is cached until the steering hook fires.
6. The steering hook adds the cached vector to the last-token residual stream:

```python
hidden[:, -1, :] = hidden[:, -1, :] + steering
```

7. The modified hidden state is returned to the rest of the model.

The `Steerer` interface is unchanged. If the returned vector shape does not match the classification activation shape, or if the cached vector cannot be added to the steering activation shape, the wrapper raises an error. If steering is enabled but no classification hook has produced a vector, the wrapper also raises an error because that should indicate an invalid hook configuration or flag combination.

This means steering is applied during normal `.forward(...)` calls and during `.generate(...)`.

---

## Hook points

`SteeredGemma2` supports three hook points.

### `layer_output`

```python
hook_point="layer_output"
```

This is the default. The hook is attached after the full decoder layer output. This is usually the most natural place to think of the activation as the post-layer residual stream.

### `post_attention`

```python
hook_point="post_attention"
```

This attaches the hook after the layer’s self-attention module.

### `post_mlp`

```python
hook_point="post_mlp"
```

This attaches the hook after the layer’s MLP module.

You can also move the hooks after initialization. The old aliases move both hooks together:

```python
steered_model.attach_hook(layer=18, hook_point="layer_output")
```

Or you can move them separately:

```python
steered_model.attach_hook(
    classify_layer=12,
    classify_hook_point="layer_output",
    steer_layer=20,
    steer_hook_point="layer_output",
)
```

---

## Turning classification and steering on and off

Classification and steering are both enabled by default.

To turn both off or on with the backward-compatible combined control:

```python
steered_model.set_enabled(False)
steered_model.set_enabled(True)
```

You can also control the two phases separately:

```python
steered_model.set_classification_enabled(True)
steered_model.set_steering_enabled(False)
```

This lets you classify and create/log steering vectors without applying them. The reverse combination, classification disabled while steering is enabled, raises an error during the forward pass because the steering hook should never fire without a cached vector.

You can temporarily disable only the steering application with a context manager:

```python
with steered_model.without_steering():
    unsteered_out = steered_model.generate(**inputs, max_new_tokens=32)

steered_out = steered_model.generate(**inputs, max_new_tokens=32)
```

This is useful for comparing steered and unsteered behavior using the same in-memory model while still exercising the classification hook.

---

## Notebook overview

The notebook in `notebooks/gemma2_benchmarks.ipynb` is intended to be run in Google Colab.

At a high level, it does the following:

1. Clones this repo from GitHub.
2. Installs the package.
3. Logs into Hugging Face.
4. Loads `google/gemma-2-2b`.
5. Creates one or more `Steerer` objects.
6. Wraps the model with `SteeredGemma2`.
7. Runs one benchmark per notebook cell.

The basic model setup looks like this:

```python
import torch
from gemma2_steering import SteeredGemma2, ZeroSteerer

steerer = ZeroSteerer()

model = SteeredGemma2(
    steerer,
    model_name="google/gemma-2-2b",
    layer=12,
    hook_point="layer_output",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

Once the wrapper is created, the same model object can be used for generation and evaluation.

---

## Benchmarking steered LLMs

The repo is designed so that the same `SteeredGemma2` object can be evaluated with `lm-evaluation-harness`.

The notebook uses one cell per benchmark. The included benchmark set is meant to cover several different behavioral axes:

* MMLU for broad multiple-choice knowledge.
* TruthfulQA for truthfulness and misconception resistance.
* A sycophancy benchmark using the local YAML task in `benchmarks/`.
* WMDP as a malicious or dangerous-knowledge style benchmark.
* Humanity’s Last Exam, if supported by the installed `lm-evaluation-harness` version.

The local sycophancy task lives here:

```text
benchmarks/sycophancy_on_nlp_survey.yaml
```

It is kept in the repo because sycophancy task availability and naming can vary across `lm-evaluation-harness` versions. Keeping the YAML local gives the repo a stable task definition that the notebook can point to directly.

The adapter in `gemma2_steering/lm_eval.py` lets the benchmark code evaluate the wrapped model rather than a fresh model loaded separately by the harness. This is important because the benchmark should measure the steered model, not an unmodified Gemma instance.

The notebook follows this general pattern:

```python
from gemma2_steering import simple_evaluate

results = simple_evaluate(
    model=steered_model,
    tasks=["truthfulqa_mc2"],
    num_fewshot=0,
    batch_size=1,
)
```

For local benchmark YAMLs, the notebook uses the benchmark include path so the harness can find repo-owned tasks.

---

## Important evaluation note

This steering hook edits only the final sequence position in each forward pass:

```python
hidden[:, -1, :]
```

For generation, this is usually the position used to predict the next token, so the intervention is directly relevant.

For log-likelihood-style benchmarks, this means the benchmark is measuring the effect of the last-token intervention under the harness’s evaluation calls. It is not doing a full activation patch across every token position in a continuation unless the evaluation call itself performs those positions as separate forward passes.

This is an intentional simplification. It keeps the steering mechanism small and easy to inspect, but it is worth remembering when interpreting benchmark results.

---

## Testing

Run tests with:

```bash
pytest -q
```

The tests use a tiny fake decoder model, not Gemma 2. They are designed to run quickly on CPU and do not require a Hugging Face token.

The tests check the main steering behaviors:

* invalid steerers are rejected;
* `ZeroSteerer` preserves output shape;
* `ConstantSteerer` enforces batched vectors;
* steering changes the selected last-token activation;
* disabling classification and steering works;
* classification and steering can use separate hooks;
* invalid classify-after-steer hook configurations are rejected;
* the hooks can be attached without loading Gemma 2.
