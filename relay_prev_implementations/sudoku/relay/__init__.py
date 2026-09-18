"""relay (sudoku): Learned Relay Representations for diffusion-style MLM.

Components used by the paper's Sudoku experiments:
- Model architecture (model.py): plain MLM and the relay variant
- Loss function (loss.py): MLM and the multi-step relay rollout (with optional BPTT)
- Predictor for inference (predictor.py)
- Data module (datamodule.py)
- Metrics computation (metrics.py)
- Type definitions (types.py)
- History tracking (history.py)
"""

from .model import RotaryTransformerModel, RotaryTransformerRelayModel
from .loss import MLMLoss, RelayBPTTLoss
from .streaming_batch import StreamingBatch
from .predictor import ConfidenceBasedPredictor
from .datamodule import (
    PuzzleInfillPredCollator,
    PuzzleMLMCollator,
)
from .types import (
    MLMBatch,
    MLMSeq2SeqPredictionBatch,
    MLMUncondtionalPredictionBatch,
    MLMLossDict,
    MLMModel,
    MLMPredictionDict,
)
from .history import HistoryTopKPlugin
from .callbacks import ValidationThresholdSweepCallback

__all__ = [
    "RotaryTransformerModel",
    "RotaryTransformerRelayModel",
    "MLMLoss",
    "RelayBPTTLoss",
    "StreamingBatch",
    "ConfidenceBasedPredictor",
    "PuzzleInfillPredCollator",
    "PuzzleMLMCollator",
    "MLMBatch",
    "MLMSeq2SeqPredictionBatch",
    "MLMUncondtionalPredictionBatch",
    "MLMLossDict",
    "MLMModel",
    "MLMPredictionDict",
    "HistoryTopKPlugin",
    "ValidationThresholdSweepCallback",
]
