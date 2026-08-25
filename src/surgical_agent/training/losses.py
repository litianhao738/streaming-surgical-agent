"""P2 smoke-only task-wise masked losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from surgical_agent.data.schemas import FrameSupervisionTarget
from surgical_agent.models.baseline import TASK_CLASS_COUNTS, FrameLogits


@dataclass(frozen=True)
class MaskedLossResult:
    """Finite aggregate plus per-task support for audit artifacts."""

    total: Tensor
    task_losses: dict[str, Tensor]
    valid_counts: dict[str, int]
    semantics: str = "SMOKE_ONLY_NOT_GATE_ERROR"


def _multihot(
    targets: tuple[FrameSupervisionTarget | None, ...],
    *,
    id_field: str,
    mask_field: str,
    class_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values = torch.zeros((len(targets), class_count), dtype=torch.float32, device=device)
    mask = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, target in enumerate(targets):
        if target is None or not getattr(target.mask, mask_field):
            continue
        mask[index] = True
        for class_id in getattr(target, id_field):
            if not 0 <= class_id < class_count:
                raise ValueError(f"{id_field} contains out-of-range ID {class_id}")
            values[index, class_id] = 1.0
    return values, mask


def _masked_mean(per_sample: Tensor, mask: Tensor, reference: Tensor) -> tuple[Tensor, int]:
    count = int(mask.sum().item())
    if count == 0:
        return reference.sum() * 0.0, 0
    return (per_sample * mask.to(per_sample.dtype)).sum() / count, count


def masked_multitask_loss(
    logits: FrameLogits,
    targets: tuple[FrameSupervisionTarget | None, ...],
) -> MaskedLossResult:
    """Normalize each task independently over samples with valid supervision."""

    if logits.batch_size != len(targets):
        raise ValueError("Logit batch size and target count differ")
    device = logits.instrument.device
    task_losses: dict[str, Tensor] = {}
    valid_counts: dict[str, int] = {}
    specs = {
        "instrument": (logits.instrument, "instrument_ids", "instrument"),
        "verb": (logits.verb, "verb_ids", "verb"),
        "target": (logits.target, "target_ids", "target"),
        "ivt": (logits.ivt, "triplet_ids", "ivt"),
    }
    for task, (task_logits, id_field, mask_field) in specs.items():
        labels, mask = _multihot(
            targets,
            id_field=id_field,
            mask_field=mask_field,
            class_count=TASK_CLASS_COUNTS[task],
            device=device,
        )
        per_sample = functional.binary_cross_entropy_with_logits(
            task_logits,
            labels,
            reduction="none",
        ).mean(dim=1)
        task_losses[task], valid_counts[task] = _masked_mean(
            per_sample,
            mask,
            task_logits,
        )

    phase_labels = torch.zeros(len(targets), dtype=torch.long, device=device)
    phase_mask = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, target in enumerate(targets):
        if target is None or not target.mask.phase:
            continue
        if target.phase_id is None or not 0 <= target.phase_id < TASK_CLASS_COUNTS["phase"]:
            raise ValueError("Enabled phase target has an invalid phase_id")
        phase_labels[index] = target.phase_id
        phase_mask[index] = True
    phase_per_sample = functional.cross_entropy(
        logits.phase,
        phase_labels,
        reduction="none",
    )
    task_losses["phase"], valid_counts["phase"] = _masked_mean(
        phase_per_sample,
        phase_mask,
        logits.phase,
    )
    total = torch.stack(tuple(task_losses.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite P2 smoke loss")
    return MaskedLossResult(
        total=total,
        task_losses=task_losses,
        valid_counts=valid_counts,
    )
