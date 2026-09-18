from typing import Any, Dict
import torch


def exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """
    target = batch["target_ids"]
    pred = loss_dict["ids"]
    return {
        "pred": pred,
        "target": target,
        "pred_length": None,
        "target_length": None,
    }


def infill_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """

    pred = loss_dict["ids"]
    target = batch["target_ids"]
    pred_mask = batch["input_ids"] == tokenizer.mask_token_id
    return {
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }


def rollout_steps_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Extract rollout_steps from predictor output for logging."""
    return {"value": loss_dict["rollout_steps"]}


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for mean loss metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing loss (since we don't use batch_loss for ARLM).

    Returns:
        Dictionary with mean loss value.
    """
    return {
        "value": loss_dict["loss"],
    }
