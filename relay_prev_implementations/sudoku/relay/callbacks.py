from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from lightning import Callback

from xlm.utils.imports import get_function


class ValidationThresholdSweepCallback(Callback):
    """Run extra val-time prediction sweeps across confidence thresholds.

    This callback evaluates predictor robustness at multiple thresholds during each
    validation epoch (e.g., every val_check_interval steps) and logs aggregated
    metrics under a separate namespace, leaving the main validation metric untouched.

    `extra_metrics` is an optional task-specific extension hook. Each entry maps a
    short metric name (e.g. ``legal_rate``) to a callable (or its dotted import path)
    that takes the predictor's ``ids`` tensor of shape ``(B, L)`` and returns a
    1-D tensor of per-sample values of shape ``(B,)`` (typically 0/1 floats). The
    mean of those values is then logged under ``{metric_prefix}/t_{tag}/{name}``
    alongside the standard metrics.
    """

    def __init__(
        self,
        thresholds: List[float],
        metric_prefix: str = "val_sweep",
        prediction_dataloader_substring: str = "prediction",
        extra_metrics: Optional[Dict[str, Union[str, Callable]]] = None,
    ):
        if not thresholds:
            raise ValueError("thresholds must be a non-empty list.")
        self.thresholds = [float(t) for t in thresholds]
        self.metric_prefix = metric_prefix
        self.prediction_dataloader_substring = prediction_dataloader_substring
        self.extra_metric_fns: Dict[str, Callable] = {}
        for name, fn in (extra_metrics or {}).items():
            self.extra_metric_fns[name] = (
                get_function(fn) if isinstance(fn, str) else fn
            )
        self._stats: Dict[float, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _to_tag(threshold: float) -> str:
        return f"{threshold:.2f}".replace(".", "p")

    def _should_run_for_dataloader(self, pl_module: torch.nn.Module, dataloader_idx: int) -> bool:
        dl_name = pl_module.dataloader_names["val"].get(dataloader_idx, "")
        return self.prediction_dataloader_substring in dl_name

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        device = pl_module.device
        self._stats = {}
        for threshold in self.thresholds:
            stats = {
                "exact_sum": torch.tensor(0.0, device=device),
                "token_correct_sum": torch.tensor(0.0, device=device),
                "token_total_sum": torch.tensor(0.0, device=device),
                "rollout_sum": torch.tensor(0.0, device=device),
                "sample_count": torch.tensor(0.0, device=device),
            }
            for name in self.extra_metric_fns:
                stats[f"extra/{name}_sum"] = torch.tensor(0.0, device=device)
            self._stats[threshold] = stats

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._should_run_for_dataloader(pl_module, dataloader_idx):
            return
        predictor = pl_module.predictor
        if not hasattr(predictor, "threshold"):
            return
        if "input_ids" not in batch or "target_ids" not in batch:
            return

        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        mask_token_id = pl_module.tokenizer.mask_token_id
        batch_size = float(target_ids.shape[0])
        dataloader_name = pl_module.dataloader_names["val"].get(dataloader_idx, "prediction")

        original_threshold = predictor.threshold
        original_diag_flag = getattr(predictor, "log_rollout_diagnostics", None)
        if original_diag_flag is not None:
            predictor.log_rollout_diagnostics = False
        try:
            for threshold in self.thresholds:
                predictor.threshold = threshold
                preds = predictor.predict(
                    batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    dataloader_name=dataloader_name,
                )
                pred_ids = preds["ids"]
                masked_positions = input_ids == mask_token_id
                token_total = masked_positions.sum().float()
                token_correct = ((pred_ids == target_ids) & masked_positions).sum().float()
                exact_sum = (pred_ids == target_ids).all(dim=-1).float().sum()
                rollout_steps = float(preds.get("rollout_steps", 0.0))

                stats = self._stats[threshold]
                stats["exact_sum"] += exact_sum
                stats["token_correct_sum"] += token_correct
                stats["token_total_sum"] += token_total
                stats["rollout_sum"] += rollout_steps * batch_size
                stats["sample_count"] += batch_size
                for name, fn in self.extra_metric_fns.items():
                    values = fn(pred_ids).to(torch.float32).reshape(-1)
                    stats[f"extra/{name}_sum"] += values.sum()
        finally:
            predictor.threshold = original_threshold
            if original_diag_flag is not None:
                predictor.log_rollout_diagnostics = original_diag_flag

    def _sync_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for threshold in self.thresholds:
            stats = self._stats[threshold]
            exact_sum = self._sync_sum(stats["exact_sum"])
            token_correct_sum = self._sync_sum(stats["token_correct_sum"])
            token_total_sum = self._sync_sum(stats["token_total_sum"])
            rollout_sum = self._sync_sum(stats["rollout_sum"])
            sample_count = self._sync_sum(stats["sample_count"])

            sample_count_safe = sample_count.clamp_min(1.0)
            token_total_safe = token_total_sum.clamp_min(1.0)

            exact_match = exact_sum / sample_count_safe
            token_accuracy = token_correct_sum / token_total_safe
            rollout_steps = rollout_sum / sample_count_safe

            tag = self._to_tag(threshold)
            base = f"{self.metric_prefix}/t_{tag}"
            pl_module.log(
                f"{base}/exact_match",
                exact_match,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            pl_module.log(
                f"{base}/token_accuracy",
                token_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )
            pl_module.log(
                f"{base}/rollout_steps",
                rollout_steps,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
                add_dataloader_idx=False,
            )

            for name in self.extra_metric_fns:
                extra_sum = self._sync_sum(stats[f"extra/{name}_sum"])
                value = extra_sum / sample_count_safe
                pl_module.log(
                    f"{base}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )
