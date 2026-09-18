"""Sudoku-specific validation metrics.

Currently exposes a single metric: ``legal_rate`` — the fraction of predicted
9×9 boards that satisfy the three Sudoku constraints (no duplicate digits in
any row, column, or 3×3 box) AND are fully filled (no remaining mask tokens).

Intended for use as an `update_fn` for `xlm.metrics.MetricWrapper` (i.e. a
plain `MeanMetric` whose `value` is a (B,) tensor of 0/1 floats).

Token convention (`SimpleSpaceTokenizer.for_numbers(vocab_size=10)`):
    PAD=0, UNK=1, MASK=2, CLS=3, SEP=4, BOS=5, EOS=6, "0"=7, ..., "9"=16.

We map IDs in [7, 16] to digits in [0, 9]; everything else is treated as a
mask / blank cell (digit 0). A digit-0 cell never participates in duplicate
checks, but it makes the board "not fully filled" so the metric counts only
boards that have all 81 cells in {1..9} and respect the three constraints.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

FIRST_DIGIT_TOKEN_ID: int = 7
GRID_SIDE: int = 9
CONTENT_LEN: int = GRID_SIDE * GRID_SIDE


@torch.no_grad()
def sudoku_legal_mask(
    ids: Tensor,
    *,
    first_digit_token_id: int = FIRST_DIGIT_TOKEN_ID,
    grid_side: int = GRID_SIDE,
) -> Tensor:
    """Return a (B,) bool tensor: True iff the board is fully filled with digits
    1..9 and contains no duplicate digit in any row/column/3×3 box.
    """
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    B = ids.shape[0]
    L = ids.shape[-1]
    if L < CONTENT_LEN:
        raise ValueError(
            f"sudoku_legal_mask expects sequences of length >= {CONTENT_LEN}; got {L}"
        )
    ids = ids[..., :CONTENT_LEN]

    digits = ids - first_digit_token_id
    valid = (digits >= 0) & (digits <= 9)
    digits = torch.where(valid, digits, torch.zeros_like(digits))
    digits = digits.view(B, grid_side, grid_side).long()

    fully_filled = (digits >= 1).view(B, -1).all(dim=-1)

    one_hot = torch.nn.functional.one_hot(digits, num_classes=10)[..., 1:]

    row_counts = one_hot.sum(dim=2)
    col_counts = one_hot.sum(dim=1)
    box = one_hot.view(B, 3, 3, 3, 3, grid_side).sum(dim=(2, 4))

    rows_legal = (row_counts <= 1).reshape(B, -1).all(dim=-1)
    cols_legal = (col_counts <= 1).reshape(B, -1).all(dim=-1)
    boxes_legal = (box <= 1).reshape(B, -1).all(dim=-1)

    return rows_legal & cols_legal & boxes_legal & fully_filled


def sudoku_legal_rate_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update fn for `MeanMetric` consuming the predictor's final ids.

    Returns ``{"value": (B,) float tensor}`` of 0/1 indicating per-puzzle legality.
    """
    pred = loss_dict["ids"]
    legal = sudoku_legal_mask(pred).to(torch.float32)
    return {"value": legal}
