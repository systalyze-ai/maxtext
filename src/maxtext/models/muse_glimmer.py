"""Decoder layer definition for Muse Glimmer models."""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Optional

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import AttentionType, Config
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


# -----------------------------------------
# The Decoder Layer for Muse Glimmer models
# -----------------------------------------

# Three sliding-window layers then one full-attention layer, repeated. The 30B checkpoint
# declares this explicitly in `text_config.layer_types` as 13 repeats over 52 layers.
MUSE_GLIMMER_ATTENTION_PATTERN = (
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  """Get attention type based on layer ID."""
  layer_id %= len(MUSE_GLIMMER_ATTENTION_PATTERN)
  return MUSE_GLIMMER_ATTENTION_PATTERN[layer_id]


def is_nope_layer(layer_id):
  """Full-attention layers carry no position embedding.

  HF encodes this as `layer_rope_theta[i] == 0` and skips the rotary for those layers
  (`position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None`).
  The zeros land exactly on the full-attention positions of the cycle.
  """
  return get_attention_type(layer_id) is attentions.AttentionType.GLOBAL


class MuseGlimmerDecoderLayer(nnx.Module):
  """Transformer decoder layer for Muse Glimmer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      attention_type: AttentionType,
      nope_layer: bool = False,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.attention_type = attention_type
    self.nope_layer = nope_layer
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    # HF wraps both the attention and the MLP in a pre/post norm pair (Gemma2 sandwich):
    #   input_layernorm(eps=rms_norm_eps) -> attn -> post_attention_layernorm(eps=post_norm_eps)
    #   pre_feedforward_layernorm(eps=rms_norm_eps) -> mlp -> post_feedforward_layernorm(eps=post_norm_eps)
    # `post_norm_eps <= 0` means "same as normalization_layer_epsilon".
    post_eps = config.post_norm_eps if config.post_norm_eps > 0 else config.normalization_layer_epsilon

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        scale_offset=1.0,
        rngs=rngs,
    )

    self.pre_mlp_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        scale_offset=1.0,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=post_eps,
        scale_offset=1.0,
        rngs=rngs,
    )

    self.post_mlp_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=post_eps,
        scale_offset=1.0,
        rngs=rngs,
    )

    # HF multiplies the QK-normed query by `qk_scale_factor` and *also* keeps the standard
    # `scaling = head_dim**-0.5` in the attention interface (it inherits AfmoeAttention's
    # __init__), so the two compose. MaxText's query_pre_attn_scalar replaces the depth
    # scaling outright, so fold both factors into it.
    query_pre_attn_scalar = (
        config.qk_scale_factor * config.head_dim**-0.5 if config.qk_scale_factor > 0 else config.head_dim**-0.5
    )

    self.attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_bias_in_projections=config.attention_bias,
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        use_qk_norm=config.use_qk_norm,
        is_nope_layer=self.nope_layer,
        rope_type=config.rope_type.lower(),
        rngs=rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.dropout = linears.Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    attention_lnx, kv_cache = self.attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    # Normalize stream before addition (post-norm placement, as in Gemma2/Olmo3).
    attention_lnx = self.post_self_attention_layer_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    intermediate_inputs = inputs + attention_lnx

    mlp_in = self.pre_mlp_layer_norm(intermediate_inputs)
    mlp_in = nn.with_logical_constraint(mlp_in, ("activation_batch", "activation_norm_length", "activation_embed"))

    mlp_lnx = self.mlp(mlp_in)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    mlp_lnx = self.post_mlp_layer_norm(mlp_lnx)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if is_scan_carry:
      stacked_kv_cache = jax.tree_util.tree_map(
          lambda stacked, updated: stacked.at[layer_idx].set(updated), stacked_kv_cache, kv_cache
      )
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    elif self.config.scan_layers:
      return layer_output, None
    else:
      # The unscanned driver unpacks two values, and the KV cache must be
      # propagated here rather than dropped.
      return layer_output, kv_cache


class MuseGlimmerScannableBlock(nnx.Module):
  """One full attention cycle (3 sliding + 1 full) as a single scannable unit.

  Scanning needs every iteration to have identical structure, but Muse Glimmer's layers
  differ within a cycle. Scanning the whole cycle instead of a single layer keeps the
  structure homogeneous, matching how Olmo3 and GPT-OSS handle the same problem.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    for layer_id in range(config.inhomogeneous_layer_cycle_interval):
      layer = MuseGlimmerDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          attention_type=get_attention_type(layer_id),
          nope_layer=is_nope_layer(layer_id),
          quant=self.quant,
          rngs=rngs,
      )
      setattr(self, f"layers_{layer_id}", layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs
    for layer_id in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layers_{layer_id}")
      y = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
      if cfg.scan_layers:
        y = y[0]
    if cfg.scan_layers:
      return y, None
    return y


MuseGlimmerScannableBlockToLinen = nnx_wrappers.to_linen_class(
    MuseGlimmerScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
