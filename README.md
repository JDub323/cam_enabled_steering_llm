# Gemma 2 Steering Research

A small repo for residual-stream steering experiments on `google/gemma-2-2b` with Hugging Face Transformers and `lm-evaluation-harness`.

The core idea is intentionally narrow: a `SteeredGemma2` wraps Gemma 2, attaches a PyTorch forward hook at a configurable decoder-layer point, sends only the batched last-token residual stream vector to a `Steerer`, and adds the returned batched steering vector back to that last-token position.

## Layout

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

## Install

```bash
pip install -e ".[eval,dev]"
```

Gemma 2 is gated on Hugging Face. Log in before loading the model:

```python
from huggingface_hub import login
login()
```

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

## Steering interface

All steerers inherit from `Steerer` and implement exactly one required method:

```python
get_steering_vector(activation: torch.Tensor) -> torch.Tensor
```

The input and output must both be batched tensors with shape `[batch, hidden_size]`. Unbatched vectors are intentionally rejected. This keeps the hook behavior explicit and makes batched generation/evaluation less error-prone.

Implemented steerers:

- `ZeroSteerer`: no-op baseline.
- `ConstantSteerer`: adds a fixed batched vector, shape `[1, hidden_size]` or `[batch, hidden_size]`.

## Hook points

`SteeredGemma2(..., hook_point=...)` supports:

- `layer_output` — default; after the full decoder layer output.
- `post_attention` — after the layer's self-attention module.
- `post_mlp` — after the layer's MLP module.

Use `model.set_enabled(False)` or `with model.without_steering(): ...` to temporarily disable steering without loading a second Gemma.

## lm-evaluation-harness

The notebook uses the programmatic `lm_eval.simple_evaluate` API. The adapter in `gemma2_steering/lm_eval.py` hands the underlying hooked Transformers model to the harness, so both log-likelihood and generation-style tasks use the same in-memory Gemma instance.

A local sycophancy task YAML is included under `benchmarks/` and loaded via `TaskManager(include_path="benchmarks")`.

Notes:

- `truthfulqa_mc2` is the recommended TruthfulQA multiple-choice task.
- `wmdp` is included as a malicious/dangerous-knowledge style benchmark.
- Humanity's Last Exam support varies by `lm-eval` release and may not be available as a built-in task. The notebook includes a cell that tries common task names and prints the available-task command if none are found.
- Because this hook only edits the last sequence position in each forward pass, log-likelihood benchmarks are measuring that exact last-token intervention rather than a full-token-by-token activation patch over every continuation token.

## Tests

```bash
pytest -q
```

The tests use a tiny fake decoder model, not Gemma 2, so they run quickly on CPU and do not require a Hugging Face token.
