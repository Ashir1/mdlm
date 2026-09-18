from typing import Any, Dict, List, NotRequired, Optional, Protocol, Tuple, TypedDict, Union

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class MLMBatch(TypedDict, total=False):
    """Input to the MLM. Different collators produce different keys.

    From-solver double-backprop:
        trajectory_prefix, x_i, x_i_plus_1, x_i_plus_2, i, target_ids
    Standard MLM:
        input_ids, attention_mask, target_ids
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Optional[Integer[TT, " batch seq_len"]]
    fixed: Bool[TT, " batch seq_len"]  # 1 = given clue (never change), 0 = mutable
    trajectory_prefix: List[Integer[TT, " batch seq_len"]]
    x_i: Integer[TT, " batch seq_len"]
    x_i_plus_1: Integer[TT, " batch seq_len"]
    x_i_plus_2: Integer[TT, " batch seq_len"]
    i: int


class MLMSeq2SeqPredictionBatch(TypedDict):
    """Input to the MLM for predicting suffix given the prefix."""

    input_ids: Integer[TT, " batch prefix_seq_len"]  # left-padded
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    target_ids: Integer[TT, " batch suffix_seq_len"]


class MLMUncondtionalPredictionBatch:
    """Input to the MLM for unconditional generation.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model. All masks.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]


class MLMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """

    loss: Float[TT, ""]


class MLMModel(Protocol):
    def __call__(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        z: Float[TT, " batch d_model"],
    ) -> Tuple[Float[TT, " batch d_model"], Float[TT, " batch seq_len vocab_size"]]: ...


class MLMPredictionDict(TypedDict):
    """Output of the Predictor for MLM.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        time_taken (List[float]): Time taken for each prediction.
        output_start_idx (Integer[TT, " batch"]): The index of the first token in the output.
    """

    loss: Optional[Float[TT, ""]]
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
    time_taken: List[float]
    output_start_idx: int
    rollout_steps: float
    rollout_diagnostics: Dict[str, Any]
    # Set when predictor.log_rollout_diagnostics; used for jsonl ``inference_steps``.
    rollout_steps_per_sample: NotRequired[List[int]]
    trajectory: NotRequired[List[List[List[int]]]]
