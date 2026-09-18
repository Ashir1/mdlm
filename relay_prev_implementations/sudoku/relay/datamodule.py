import torch
from xlm.datamodule import (
    Collator,
    BaseCollatorInput,
    Tokenizer,
)
from jaxtyping import Float, Integer
from torch import Tensor as TT
from typing import Callable, Dict, List, Literal, Optional, Any
from .types import MLMBatch
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class PuzzleMLMCollator(Collator):
    """Training/validation collator for puzzle infill tasks.

    When ``mask_all=False`` (default) each non-fixed position is masked
    independently with probability ``t ~ Uniform[0, 1]`` — a standard uniform
    MLM noise schedule.  When ``mask_all=True`` every non-fixed position is
    masked unconditionally, giving the fully-masked rollout-start distribution
    used by the relay / BPTT experiments.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        mask_all: bool = False,
    ):
        self.tokenizer = tokenizer
        self.mask_all = mask_all

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MLMBatch:
        batch_size = len(examples)
        t: Float[TT, " batch_size"] = torch.rand(batch_size)

        prompt_ids = torch.tensor([e["prompt_ids"] for e in examples], dtype=torch.long)
        fixed = (prompt_ids != self.tokenizer.mask_token_id).to(torch.bool)

        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        target_ids = input_ids.clone()
        if self.mask_all:
            mask = torch.ones_like(input_ids, dtype=torch.bool) & ~fixed
        else:
            mask = (torch.rand_like(input_ids, dtype=t.dtype) < t[:, None]) & ~fixed
        input_ids[mask] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "fixed": fixed,
        }


class PuzzleInfillPredCollator(Collator):
    """Prediction collator for puzzle infill tasks.

    Presents each example as a fully-masked prompt (``input_ids = prompt_ids``,
    fixed clue positions preserved) paired with the ground-truth solution as
    ``target_ids``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MLMBatch:
        
        input_ids = torch.tensor([e["prompt_ids"] for e in examples], dtype=torch.long)
        target_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        fixed = (input_ids != self.tokenizer.mask_token_id).to(torch.bool)
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "fixed": fixed,
        }


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer):
    _ids = ids.clone()
    _ids[_ids == -100] = tokenizer.pad_token_id
    return _ids


def print_batch(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging MLM batches.

    Args:
        batch: The batch to print.
        split: The split name.
        tokenizer: The tokenizer to decode tokens.
        dataloader_name: Name of the dataloader.
    """
    logger.info(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    _input_ids = _replace_100_with_pad(batch["input_ids"][0], tokenizer)
    print(tokenizer.decode(_input_ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    if "attention_mask" in batch:
        print("attention_mask (int):")
        print(batch["attention_mask"][0].int())
    print("target_ids:")
    print(batch["target_ids"][0])
    print("target tokens:")
    _target_ids = _replace_100_with_pad(batch["target_ids"][0], tokenizer)
    print(tokenizer.decode(_target_ids))
