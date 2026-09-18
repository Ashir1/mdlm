"""Vendored Fast-dLLM v2 model with the RELAY extension (paper Algorithm 1).

Importing this package registers ``Fast_dLLM_QwenConfig`` and
``Fast_dLLM_QwenForCausalLM`` with the Hugging Face Auto* classes so that
``AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=False)`` resolves
to the local class instead of downloading remote code.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import Fast_dLLM_QwenConfig
from .modeling import (
    Fast_dLLM_QwenForCausalLM,
    Fast_dLLM_QwenModel,
)

AutoConfig.register(Fast_dLLM_QwenConfig.model_type, Fast_dLLM_QwenConfig)
AutoModelForCausalLM.register(Fast_dLLM_QwenConfig, Fast_dLLM_QwenForCausalLM)

__all__ = [
    "Fast_dLLM_QwenConfig",
    "Fast_dLLM_QwenForCausalLM",
    "Fast_dLLM_QwenModel",
]
