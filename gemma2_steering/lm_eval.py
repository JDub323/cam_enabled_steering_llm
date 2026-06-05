from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .model import SteeredGemma2


def as_lm_eval_model(steered: SteeredGemma2, **hf_lm_kwargs: Any) -> Any:
    """Wrap the hooked HF model for lm-evaluation-harness.

    The harness sees the underlying Transformers model, but the hook remains
    installed because SteeredGemma2 registered it on that model in-place.
    """
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=steered.model,
        tokenizer=hf_lm_kwargs.pop("tokenizer", steered.tokenizer),
        backend=hf_lm_kwargs.pop("backend", "causal"),
        **hf_lm_kwargs,
    )


def simple_evaluate(
    steered: SteeredGemma2,
    tasks: str | Sequence[str],
    *,
    include_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run lm-eval on a steered model using the programmatic API."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    task_manager = TaskManager(include_path=include_path) if include_path else None
    lm = as_lm_eval_model(steered, **kwargs.pop("hf_lm_kwargs", {}))
    return lm_eval.simple_evaluate(model=lm, tasks=task_list, task_manager=task_manager, **kwargs)
