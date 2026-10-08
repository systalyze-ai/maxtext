# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests comparing the MaxText Muse Glimmer text model against the Hugging Face reference.

The reference is the `muse_glimmer` module in transformers 5.15 and later. The parity tests run on
CPU in float32 with random weights at small dimensions, and need no network and no checkpoint.
The config checks also read `config.json` and `model.safetensors.index.json` of
meta-models/Muse-Glimmer-30B from the local Hugging Face cache when they are there.
"""

import contextlib
import copy
import json
import unittest
from unittest import mock

from flax import nnx
from huggingface_hub import try_to_load_from_cache
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.utils import param_key_parts_from_path, process_maxtext_param
from maxtext.checkpoint_conversion.utils.utils import validate_and_filter_param_map_keys
from maxtext.common.common_types import AttentionType, DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import models, muse_glimmer
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import HF_IDS
from tests.utils.test_helpers import get_test_config_path

try:
  import torch
  import transformers
  from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
  from transformers.models.muse_glimmer import modeling_muse_glimmer as hf_glimmer

  HAS_REFERENCE = True
except ImportError:
  HAS_REFERENCE = False

pytestmark = [pytest.mark.cpu_only]

needs_reference = pytest.mark.skipif(not HAS_REFERENCE, reason="needs torch and transformers>=5.15 (Muse Glimmer)")

_MODEL_NAME = "muse-glimmer-30b"
_TEXT_PREFIXES = ("model.language_model.", "lm_head.")

# Small dimensions that keep GQA and two (sliding, sliding, sliding, full) cycles, with a
# window the sequence crosses several times.
_LAYERS = 8
_EMB = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16
_MLP = 96
_VOCAB = 128
_WINDOW = 4
_BATCH = 2
_SEQ = 24
_PREFILL = 8

_LAYER_TOL = {"rtol": 1e-5, "atol": 1e-5}
_LOGIT_TOL = {"rtol": 1e-4, "atol": 1e-3}
# A negative control must change the output by at least this multiple of the matching tolerance.
_CONTROL_MARGIN = 100


def _hub_file(filename):
  """Returns the path of a Muse Glimmer hub file in the local HF cache, or None."""
  path = try_to_load_from_cache(HF_IDS[_MODEL_NAME], filename)
  return path if isinstance(path, str) else None


def _hf_config(**text_overrides):
  """The released config with the text model shrunk and a minimal vision tower."""
  config = copy.deepcopy(hf_model_configs.muse_glimmer_30b_dict)
  text = config["text_config"]
  text.update(
      hidden_size=_EMB,
      intermediate_size=_MLP,
      num_hidden_layers=_LAYERS,
      num_attention_heads=_HEADS,
      num_key_value_heads=_KV_HEADS,
      head_dim=_HEAD_DIM,
      vocab_size=_VOCAB,
      sliding_window=_WINDOW,
      layer_types=text["layer_types"][:_LAYERS],
      layer_rope_theta=text["layer_rope_theta"][:_LAYERS],
  )
  text.update(text_overrides)
  config["vision_config"].update(
      hidden_size=8, intermediate_size=16, num_attention_heads=2, num_hidden_layers=1, layer_types=["full_attention"]
  )
  config.update(out_hidden_size=32, projector_hidden_size=16)
  config = transformers.MuseGlimmerConfig(**config)
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  config.text_config._attn_implementation = "eager"  # pylint: disable=protected-access
  return config


def _maxtext_config(**overrides):
  """The muse-glimmer-30b config at the same small dimensions, in float32 on CPU."""
  kwargs = {
      "model_name": _MODEL_NAME,
      "override_model_config": True,
      "run_name": "muse_glimmer_vs_reference_test",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "base_emb_dim": _EMB,
      "base_num_query_heads": _HEADS,
      "base_num_kv_heads": _KV_HEADS,
      "head_dim": _HEAD_DIM,
      "base_mlp_dim": _MLP,
      "base_num_decoder_layers": _LAYERS,
      "vocab_size": _VOCAB,
      "sliding_window_size": _WINDOW,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": _BATCH,
      "scan_layers": False,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_qk_product": True,
      "float32_logits": True,
      "dropout_rate": 0.0,
  }
  kwargs.update(overrides)
  return pyconfig.initialize(["", get_test_config_path()], **kwargs)


def _randomize(module, seed):
  """Gives every parameter seeded random values, so no mapping is checked against zeros or ones."""
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, param in module.named_parameters():
      normal = torch.randn(param.shape, generator=gen)
      if name.endswith("layernorm.weight"):
        # Centered norms, applied as (1 + w).
        value = 0.3 * normal
      elif name.endswith("norm.weight"):
        # The final norm, applied as w.
        value = 1.0 + 0.3 * normal
      elif name.endswith("embed_tokens.weight"):
        # Row RMS from 0.1 to 10, so a missing embedding norm shows.
        value = normal * torch.empty(param.shape[0], 1).uniform_(0.1, 10.0, generator=gen)
      elif name.endswith(("o_proj.weight", "down_proj.weight")):
        # Small sublayer outputs, so post_norm_eps (1e-8) and rms_norm_eps (1e-5) give different results.
        value = 0.1 * normal / param.shape[-1] ** 0.5
      elif name == "lm_head.weight":
        # Logits large enough to saturate the softcap.
        value = 60.0 * normal / param.shape[-1] ** 0.5
      else:
        value = normal / param.shape[-1] ** 0.5
      param.copy_(value)


def _load_nnx(module, mt_prefix, hf_state, mapping, hooks):
  """Loads the mapped HF tensors under `mt_prefix` into an NNX module through the conversion hooks."""
  loaded = 0
  for mt_key, hf_key in mapping.items():
    if not mt_key.startswith(mt_prefix):
      continue
    *path, leaf = mt_key[len(mt_prefix) :].split("-")
    owner = module
    for name in path:
      owner = getattr(owner, name)
    param = getattr(owner, leaf)
    param.value = jnp.asarray(hooks[mt_key](hf_state[hf_key], param.value.shape))
    loaded += 1
  return loaded


class _ParityTestCase(unittest.TestCase):

  def assert_differs(self, actual, reference, tol, what):
    """Negative control: `reference` is a variant the tested output must not match."""
    diff = float(np.max(np.abs(np.asarray(actual, np.float64) - np.asarray(reference, np.float64))))
    self.assertGreater(diff, _CONTROL_MARGIN * tol["atol"], f"output is insensitive to {what}")


@needs_reference
@pytest.mark.scheduled_only
class MuseGlimmerLayerTest(_ParityTestCase):
  """Attention and decoder-layer parity for a sliding RoPE layer and a global NoPE layer."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch.set_grad_enabled(False)
    cls.mt_cfg = _maxtext_config()
    cls.hf_cfg = _hf_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.mt_cfg), cls.mt_cfg.mesh_axes)
    hf_dict = cls.hf_cfg.to_dict()
    cls.mapping = PARAM_MAPPING[_MODEL_NAME](hf_dict, cls.mt_cfg, scan_layers=False)
    cls.hooks = HOOK_FNS[_MODEL_NAME](hf_dict, cls.mt_cfg, scan_layers=False, saving_to_hf=False)
    rng = np.random.default_rng(0)
    cls.x = rng.standard_normal((_BATCH, _SEQ, _EMB), dtype=np.float32)
    cls.positions = np.broadcast_to(np.arange(_SEQ, dtype=np.int32), (_BATCH, _SEQ))

  def _layers(self, layer_idx):
    """An HF decoder layer with random weights, and a MaxText layer loaded from it through the mapping."""
    hf_layer = hf_glimmer.MuseGlimmerTextDecoderLayer(self.hf_cfg.text_config, layer_idx).eval()
    _randomize(hf_layer, seed=layer_idx)
    mt_layer = muse_glimmer.MuseGlimmerDecoderLayer(
        config=self.mt_cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        attention_type=muse_glimmer.get_attention_type(layer_idx),
        nope_layer=muse_glimmer.is_nope_layer(layer_idx),
        rngs=nnx.Rngs(0),
    )
    hf_state = {f"model.language_model.layers.{layer_idx}.{k}": v.numpy() for k, v in hf_layer.state_dict().items()}
    loaded = _load_nnx(mt_layer, f"params-decoder-layers_{layer_idx}-", hf_state, self.mapping, self.hooks)
    self.assertEqual(loaded, len(hf_state))
    self.assertEqual(loaded, len(jax.tree.leaves(nnx.state(mt_layer, nnx.Param))))
    return hf_layer, mt_layer

  def _hf_inputs(self, layer_idx, text_config=None, rope=None):
    """The hidden states, mask and rotary tables that HF's text model passes to layer `layer_idx`."""
    text_config = text_config or self.hf_cfg.text_config
    x = torch.from_numpy(self.x)
    positions = torch.from_numpy(self.positions).long()
    sliding = text_config.layer_types[layer_idx] == "sliding_attention"
    mask = (create_sliding_window_causal_mask if sliding else create_causal_mask)(
        config=text_config,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=positions,
        allow_is_causal_skip=False,
    )
    if rope is None:
      rope = bool(text_config.layer_rope_theta[layer_idx])
    position_embeddings = hf_glimmer.MuseGlimmerTextRotaryEmbedding(text_config)(x, positions) if rope else None
    return x, mask, position_embeddings

  def _hf_attention(self, hf_layer, layer_idx, **kwargs):
    x, mask, position_embeddings = self._hf_inputs(layer_idx, **kwargs)
    return hf_layer.self_attn(hidden_states=x, position_embeddings=position_embeddings, attention_mask=mask)[0].numpy()

  def _hf_decoder_layer(self, hf_layer, layer_idx):
    x, mask, position_embeddings = self._hf_inputs(layer_idx)
    return hf_layer(x, position_embeddings=position_embeddings, attention_mask=mask).numpy()

  def _mt_attention(self, mt_layer):
    """Runs the attention of a MaxText layer on the shared inputs."""
    x = jnp.asarray(self.x)
    out, _ = mt_layer.attention(
        x,
        x,
        jnp.asarray(self.positions),
        decoder_segment_ids=jnp.ones((_BATCH, _SEQ), jnp.int32),
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return np.asarray(out)

  def _mt_decoder_layer(self, mt_layer):
    out, _ = mt_layer(
        jnp.asarray(self.x), jnp.ones((_BATCH, _SEQ), jnp.int32), jnp.asarray(self.positions), True, MODEL_MODE_TRAIN
    )
    return np.asarray(out)

  def test_sliding_attention_matches_hf_across_the_window(self):
    hf_layer, mt_layer = self._layers(0)
    self.assertEqual(mt_layer.attention.attention_type, AttentionType.LOCAL_SLIDING)
    mt_out = self._mt_attention(mt_layer)
    np.testing.assert_allclose(mt_out, self._hf_attention(hf_layer, 0), **_LAYER_TOL)

    for window in (_WINDOW - 1, _WINDOW + 1):
      other = _hf_config(sliding_window=window).text_config
      self.assert_differs(mt_out, self._hf_attention(hf_layer, 0, text_config=other), _LAYER_TOL, f"window {window}")
    self.assert_differs(mt_out, self._hf_attention(hf_layer, 0, rope=False), _LAYER_TOL, "dropping RoPE")

  def test_global_attention_is_full_causal_and_nope(self):
    hf_layer, mt_layer = self._layers(3)
    self.assertEqual(mt_layer.attention.attention_type, AttentionType.GLOBAL)
    self.assertTrue(mt_layer.attention.is_nope_layer)
    mt_out = self._mt_attention(mt_layer)
    np.testing.assert_allclose(mt_out, self._hf_attention(hf_layer, 3), **_LAYER_TOL)

    self.assert_differs(mt_out, self._hf_attention(hf_layer, 3, rope=True), _LAYER_TOL, "RoPE on a NoPE layer")
    windowed = _hf_config(layer_types=["sliding_attention"] * _LAYERS).text_config
    self.assert_differs(mt_out, self._hf_attention(hf_layer, 3, text_config=windowed), _LAYER_TOL, "a window")

  def test_output_gate_and_query_scale(self):
    hf_layer, mt_layer = self._layers(1)
    hf_out = self._hf_attention(hf_layer, 1)
    np.testing.assert_allclose(self._mt_attention(mt_layer), hf_out, **_LAYER_TOL)

    attention = mt_layer.attention
    with mock.patch.object(attention, "attn_gate", None):
      self.assert_differs(self._mt_attention(mt_layer), hf_out, _LAYER_TOL, "the output gate")
    # HF applies qk_scale_factor on top of head_dim**-0.5; either factor alone is wrong.
    for scale in (self.mt_cfg.qk_scale_factor, self.mt_cfg.head_dim**-0.5):
      with mock.patch.object(attention, "query_pre_attn_scalar", scale):
        self.assert_differs(self._mt_attention(mt_layer), hf_out, _LAYER_TOL, f"query scale {scale}")

  def test_rope_matches_hf_up_to_max_position_embeddings(self):
    # The released head_dim and theta. RoPE is relative, so large absolute positions only change the
    # rounding, and that rounding must match HF.
    head_dim = hf_model_configs.muse_glimmer_30b_dict["text_config"]["head_dim"]
    text_config = _hf_config(head_dim=head_dim).text_config
    mt_layer = muse_glimmer.MuseGlimmerDecoderLayer(
        config=_maxtext_config(head_dim=head_dim),
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        attention_type=AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    positions = np.linspace(0, text_config.max_position_embeddings - 1, 257).astype(np.int64)[None]
    x = np.random.default_rng(1).standard_normal((1, positions.shape[1], _HEADS, head_dim), dtype=np.float32)
    mt_out = np.asarray(
        mt_layer.attention.apply_rotary_embedding(jnp.asarray(x), inputs_positions=jnp.asarray(positions))
    )

    def hf_rope(pos):
      heads_first = torch.from_numpy(x).transpose(1, 2)
      cos, sin = hf_glimmer.MuseGlimmerTextRotaryEmbedding(text_config)(heads_first, torch.from_numpy(pos))
      return hf_glimmer.apply_rotary_pos_emb(heads_first, heads_first, cos, sin)[0].transpose(1, 2).numpy()

    np.testing.assert_allclose(mt_out, hf_rope(positions), **_LAYER_TOL)
    self.assert_differs(mt_out, hf_rope(positions + 1), _LAYER_TOL, "a one-position shift")

  def test_sliding_decoder_layer_matches_hf(self):
    hf_layer, mt_layer = self._layers(2)
    np.testing.assert_allclose(self._mt_decoder_layer(mt_layer), self._hf_decoder_layer(hf_layer, 2), **_LAYER_TOL)

  def test_global_decoder_layer_matches_hf(self):
    hf_layer, mt_layer = self._layers(7)
    np.testing.assert_allclose(self._mt_decoder_layer(mt_layer), self._hf_decoder_layer(hf_layer, 7), **_LAYER_TOL)

  def test_decoder_layer_norm_conventions(self):
    hf_layer, mt_layer = self._layers(0)
    hf_out = self._hf_decoder_layer(hf_layer, 0)
    np.testing.assert_allclose(self._mt_decoder_layer(mt_layer), hf_out, **_LAYER_TOL)

    post_norms = [mt_layer.post_self_attention_layer_norm, mt_layer.post_mlp_layer_norm]
    sandwich = [mt_layer.pre_self_attention_layer_norm, mt_layer.pre_mlp_layer_norm] + post_norms
    with contextlib.ExitStack() as stack:
      for norm in sandwich:
        stack.enter_context(mock.patch.object(norm, "scale_offset", 0.0))
      self.assert_differs(self._mt_decoder_layer(mt_layer), hf_out, _LAYER_TOL, "the (1 + w) norm scale")
    eps = self.mt_cfg.normalization_layer_epsilon
    with mock.patch.object(post_norms[0], "epsilon", eps), mock.patch.object(post_norms[1], "epsilon", eps):
      self.assert_differs(self._mt_decoder_layer(mt_layer), hf_out, _LAYER_TOL, "post_norm_eps")


@needs_reference
@pytest.mark.scheduled_only
class MuseGlimmerModelTest(_ParityTestCase):
  """Logits of a small multi-layer model whose weights go through PARAM_MAPPING and HOOK_FNS."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch.set_grad_enabled(False)
    cls.hf_cfg = _hf_config()
    cls.hf_model = hf_glimmer.MuseGlimmerForConditionalGeneration(cls.hf_cfg).eval()
    _randomize(cls.hf_model, seed=0)
    cls.hf_state = {k: v.numpy() for k, v in cls.hf_model.state_dict().items()}
    cls.ids = np.random.default_rng(0).integers(0, _VOCAB, size=(_BATCH, _SEQ))

  def _convert(self, cfg):
    """Runs the to_maxtext weight transformation on the in-memory HF state dict."""
    hf_dict = self.hf_cfg.to_dict()
    mapping = PARAM_MAPPING[_MODEL_NAME](hf_dict, cfg, cfg.scan_layers)
    hooks = HOOK_FNS[_MODEL_NAME](hf_dict, cfg, cfg.scan_layers, saving_to_hf=False)
    abstract, treedef = to_maxtext.get_maxtext_model_info(cfg)
    self.assertEqual(set(mapping), set(abstract))

    weights = [None] * len(abstract)
    consumed = []
    for mt_key in validate_and_filter_param_map_keys(mapping.keys(), abstract.keys()):
      hf_keys = mapping[mt_key]
      consumed += hf_keys if isinstance(hf_keys, list) else [hf_keys]
      index, shape = to_maxtext._get_maxtext_indices_and_shapes(mt_key, abstract)  # pylint: disable=protected-access
      load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
          hf_keys, self.hf_state.__getitem__, hooks.get(mt_key), shape, cfg, mt_key
      )
      to_maxtext._get_maxtext_weight(load_fn, index, shape, mt_key, weights, None, False)  # pylint: disable=protected-access
    self.assertCountEqual(consumed, [k for k in self.hf_state if k.startswith(_TEXT_PREFIXES)])
    tree = jax.tree_util.tree_unflatten(treedef, weights)
    # The abstract tree is rooted at the variable collection, as the checkpoint stores it.
    self.assertEqual(set(tree), {"params"})
    return tree["params"], mapping

  def _mt_logits(self, cfg, params):
    """Runs the full MaxText model in train mode on the shared token ids."""
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
    positions = np.broadcast_to(np.arange(_SEQ, dtype=np.int32), (_BATCH, _SEQ))
    logits = model.apply(
        {"params": params},
        jnp.asarray(self.ids),
        jnp.asarray(positions),
        jnp.ones((_BATCH, _SEQ), jnp.int32),
        enable_dropout=False,
    )
    return np.asarray(logits)

  def _hf_logits(self):
    return self.hf_model(input_ids=torch.from_numpy(self.ids)).logits.numpy()

  def _mt_decode_logits(self, cfg, params):
    """Prefills cfg.max_prefill_predict_length tokens, then decodes the rest one token at a time."""
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
    ids = jnp.asarray(self.ids)
    positions = jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32), (_BATCH, _SEQ))
    prefill = cfg.max_prefill_predict_length
    prefill_args = (ids[:, :prefill], positions[:, :prefill])
    prefill_kwargs = {
        "decoder_segment_ids": jnp.full((_BATCH, prefill), DECODING_ACTIVE_SEQUENCE_INDICATOR, jnp.int32),
        "model_mode": MODEL_MODE_PREFILL,
        "enable_dropout": False,
    }
    variables = model.init({"params": jax.random.PRNGKey(0)}, *prefill_args, **prefill_kwargs)
    variables = {**variables, "params": params}
    logits, cache = model.apply(variables, *prefill_args, **prefill_kwargs, mutable=["cache"])
    steps = [np.asarray(logits)]
    for i in range(prefill, _SEQ):
      variables.update(cache)
      logits, cache = model.apply(
          variables,
          ids[:, i : i + 1],
          positions[:, i : i + 1],
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          enable_dropout=False,
          mutable=["cache"],
      )
      steps.append(np.asarray(logits))
    return np.concatenate(steps, axis=1)

  def test_unscanned_logits_match_hf(self):
    cfg = _maxtext_config(scan_layers=False)
    params, _ = self._convert(cfg)
    mt_logits = self._mt_logits(cfg, params)
    np.testing.assert_allclose(mt_logits, self._hf_logits(), **_LOGIT_TOL)

    text_config = self.hf_model.config.text_config
    embed_tokens = self.hf_model.model.language_model.embed_tokens
    with mock.patch.object(embed_tokens, "embed_norm", torch.nn.Identity()):
      self.assert_differs(mt_logits, self._hf_logits(), _LOGIT_TOL, "the weightless embedding norm")
    with mock.patch.object(text_config, "output_multiplier", 1.0):
      self.assert_differs(mt_logits, self._hf_logits(), _LOGIT_TOL, "output_multiplier")
    with mock.patch.object(text_config, "final_logit_softcapping", 1e9):
      self.assert_differs(mt_logits, self._hf_logits(), _LOGIT_TOL, "the logit softcap")

  def test_scanned_logits_match_hf(self):
    cfg = _maxtext_config(scan_layers=True)
    params, _ = self._convert(cfg)
    np.testing.assert_allclose(self._mt_logits(cfg, params), self._hf_logits(), **_LOGIT_TOL)

  def test_decode_matches_hf_while_the_window_covers_the_sequence(self):
    cfg = _maxtext_config(sliding_window_size=_SEQ, max_prefill_predict_length=_PREFILL)
    params, _ = self._convert(cfg)
    with mock.patch.object(self.hf_model.config.text_config, "sliding_window", _SEQ):
      hf_logits = self._hf_logits()
    np.testing.assert_allclose(self._mt_decode_logits(cfg, params), hf_logits, **_LOGIT_TOL)

  @pytest.mark.xfail(
      strict=True,
      reason=(
          "Pre-existing on main, and Olmo3 fails the same way: during decode, AttentionOp.generate_attention_mask "
          "anchors the sliding window at the last slot of each KV-cache segment (next_pos = kv_seq_len - 1), "
          "not at the query position."
      ),
  )
  def test_decode_past_the_window_matches_hf(self):
    cfg = _maxtext_config(max_prefill_predict_length=_PREFILL)
    params, _ = self._convert(cfg)
    np.testing.assert_allclose(self._mt_decode_logits(cfg, params), self._hf_logits(), **_LOGIT_TOL)

  def test_saving_hooks_invert_loading(self):
    hf_dict = self.hf_cfg.to_dict()
    hf_shapes = {k: v.shape for k, v in self.hf_state.items()}
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        cfg = _maxtext_config(scan_layers=scan_layers)
        params, mapping = self._convert(cfg)
        hooks = HOOK_FNS[_MODEL_NAME](hf_dict, cfg, scan_layers, saving_to_hf=True)
        restored = {}
        for path, weight in jax.tree_util.tree_flatten_with_path(params)[0]:
          mt_key = "params-" + "-".join(param_key_parts_from_path(path))
          restored.update(process_maxtext_param(mt_key, jnp.asarray(weight), mapping, hooks, hf_shapes, cfg))
        self.assertCountEqual(restored, [k for k in self.hf_state if k.startswith(_TEXT_PREFIXES)])
        for hf_key, weight in restored.items():
          np.testing.assert_array_equal(weight, self.hf_state[hf_key], err_msg=hf_key)


class MuseGlimmerConfigTest(unittest.TestCase):
  """The model config and conversion tables against the released HF config and checkpoint index."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    path = _hub_file("config.json")
    cls.hub_config = None
    if path:
      with open(path, encoding="utf-8") as f:
        cls.hub_config = json.load(f)
    cls.hf_config = cls.hub_config or hf_model_configs.muse_glimmer_30b_dict
    cls.text = cls.hf_config["text_config"]
    cls.mt_cfg = pyconfig.initialize(
        ["", get_test_config_path()],
        model_name=_MODEL_NAME,
        run_name="muse_glimmer_config_test",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
    )

  def test_transcribed_config_matches_hub_config_json(self):
    if self.hub_config is None:
      self.skipTest("config.json of the released checkpoint is not in the local HF cache")
    self.assertEqual(hf_model_configs.muse_glimmer_30b_dict, self.hub_config)

  def test_maxtext_config_matches_hf_config(self):
    cfg, text = self.mt_cfg, self.text
    expected = {
        "emb_dim": text["hidden_size"],
        "num_decoder_layers": text["num_hidden_layers"],
        "num_query_heads": text["num_attention_heads"],
        "num_kv_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "mlp_dim": text["intermediate_size"],
        "vocab_size": text["vocab_size"],
        "normalization_layer_epsilon": text["rms_norm_eps"],
        "post_norm_eps": text["post_norm_eps"],
        "sliding_window_size": text["sliding_window"],
        "rope_type": text["rope_parameters"]["rope_type"],
        "rope_max_timescale": text["rope_parameters"]["rope_theta"],
        "qk_scale_factor": text["qk_scale_factor"],
        "final_logits_soft_cap": text["final_logit_softcapping"],
        "output_logits_multiplier": text["output_multiplier"],
        "logits_via_embedding": text["tie_word_embeddings"],
        "attention_bias": text["attention_bias"],
        "mlp_activations": [text["hidden_activation"], "linear"],
    }
    for key, value in expected.items():
      self.assertEqual(getattr(cfg, key), value, key)
    self.assertEqual(self.mt_cfg.decoder_block.value, "muse_glimmer")

  def test_layer_pattern_matches_hf_layer_types_and_rope_theta(self):
    theta = self.text["rope_parameters"]["rope_theta"]
    for layer_id, (layer_type, layer_theta) in enumerate(zip(self.text["layer_types"], self.text["layer_rope_theta"])):
      expected = AttentionType.GLOBAL if layer_type == "full_attention" else AttentionType.LOCAL_SLIDING
      self.assertEqual(muse_glimmer.get_attention_type(layer_id), expected, layer_id)
      self.assertEqual(muse_glimmer.is_nope_layer(layer_id), layer_theta == 0, layer_id)
      if layer_theta:
        self.assertEqual(layer_theta, theta, layer_id)
    self.assertEqual(self.text["num_hidden_layers"] % self.mt_cfg.inhomogeneous_layer_cycle_interval, 0)

  def test_mapping_covers_every_text_tensor_in_checkpoint_index(self):
    path = _hub_file("model.safetensors.index.json")
    if path is None:
      self.skipTest("model.safetensors.index.json of the released checkpoint is not in the local HF cache")
    with open(path, encoding="utf-8") as f:
      checkpoint_keys = set(json.load(f)["weight_map"])
    text_keys = {k for k in checkpoint_keys if k.startswith(_TEXT_PREFIXES)}
    self.assertGreater(len(text_keys), 0)
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        mapping = PARAM_MAPPING[_MODEL_NAME](self.hf_config, self.mt_cfg, scan_layers)
        mapped = []
        for hf_keys in mapping.values():
          mapped += hf_keys if isinstance(hf_keys, list) else [hf_keys]
        self.assertCountEqual(mapped, text_keys)


if __name__ == "__main__":
  unittest.main()
