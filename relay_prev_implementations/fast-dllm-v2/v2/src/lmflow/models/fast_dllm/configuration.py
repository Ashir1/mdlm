"""Fast_dLLM_Qwen model configuration with the relay extension.

Adds two fields on top of upstream Fast-dLLM v2:

- ``use_relay``: enable the relay LayerNorm injection (``x = token_emb + relay_layer_norm(h_t)``).
- ``relay_layer``: which encoder layer's hidden state is read off as the relay
  state ``h_s`` (default ``-1`` = last layer; supports negative indexing).

The previously-explored bridge variants (``use_cab`` / Cross Attention Bridge,
``use_mlp_carry`` / MLP bottleneck) and ``loophole_position_guard='mutable'``
were ablations that did not enter the paper and have been removed; only the
position-guarded LayerNorm relay path survives.
"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


_DEPRECATED_KEYS = (
    "use_cab", "cab_bottleneck_dim", "cab_n_heads", "cab_n_kv_heads",
    "cab_mlp_expansion_dim", "read_layers", "only_mask_tokens",
    "use_mlp_carry", "loophole_position_guard",
    "use_loopholing", "loophole_layer",
)


class Fast_dLLM_QwenConfig(PretrainedConfig):

    model_type = "Fast_dLLM_Qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        bd_size=32,
        mask_token_id=151665,
        # Relay extension (paper Algorithm 1).
        use_relay=False,
        relay_layer=-1,
        **kwargs,
    ):
        # Accept and warn on deprecated keys for backward compatibility with
        # checkpoints saved during the BPTT-loopguard / CAB development branch.
        legacy_use_relay = kwargs.pop("use_loopholing", None)
        legacy_relay_layer = kwargs.pop("loophole_layer", None)
        for key in _DEPRECATED_KEYS:
            if key in kwargs:
                logger.warning(
                    "Fast_dLLM_QwenConfig: deprecated field %r ignored. "
                    "Only the relay LayerNorm path is supported in the public release.",
                    key,
                )
                kwargs.pop(key)
        if legacy_use_relay is not None and not use_relay:
            use_relay = bool(legacy_use_relay)
        if legacy_relay_layer is not None and relay_layer == -1:
            relay_layer = int(legacy_relay_layer)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.bd_size = bd_size
        self.mask_token_id = mask_token_id

        self.use_relay = use_relay
        self.relay_layer = relay_layer

        # Validate rotary position embeddings parameters.
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
