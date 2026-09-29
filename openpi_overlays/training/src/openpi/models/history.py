"""NNX history encoder and shared H1/H2 compressor."""

from __future__ import annotations

import dataclasses
from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class HistoryConfig:
    variant: Literal["H1", "H2", "H3", "H4", "H5"]
    input_mode: Literal["raw", "cached"] = "cached"
    injection: Literal["prefix", "residual"] = "prefix"
    selection: Literal["early2_recent6", "recent8"] = "early2_recent6"
    max_records: int = 8
    queries_per_record: int = 2
    width: int = 512
    heads: int = 8
    mlp_hidden: int = 2048
    max_patches: int = 256
    mask_numeric: bool = False

    def __post_init__(self) -> None:
        if self.variant == "H3" and self.queries_per_record != 4:
            raise ValueError("H3 requires four queries per record (32 maximum tokens)")
        if self.variant != "H3" and self.queries_per_record != 2:
            raise ValueError("H1/H2/H4/H5 require two queries per record")
        if self.variant == "H4" and self.selection != "recent8":
            raise ValueError("H4 must select the recent eight transitions")
        if self.variant != "H4" and self.selection != "early2_recent6":
            raise ValueError("H1/H2/H3/H5 must select early two plus recent six")
        if self.variant == "H5" and not self.mask_numeric:
            raise ValueError("H5 must mask all history numeric tokens")
        if self.variant != "H5" and self.mask_numeric:
            raise ValueError("numeric masking is reserved for H5")
        if self.width % self.heads:
            raise ValueError("history width must be divisible by number of heads")

    @property
    def joint_attention(self) -> bool:
        return self.variant != "H1"

    @property
    def num_queries(self) -> int:
        return self.max_records * self.queries_per_record


def config_for_variant(
    variant: str,
    *,
    input_mode: Literal["raw", "cached"] = "cached",
    injection: Literal["prefix", "residual"] = "prefix",
) -> HistoryConfig | None:
    if variant == "H0":
        return None
    if variant == "H1":
        return HistoryConfig(variant="H1", input_mode=input_mode, injection=injection)
    if variant == "H2":
        return HistoryConfig(variant="H2", input_mode=input_mode, injection=injection)
    if variant == "H3":
        return HistoryConfig(variant="H3", input_mode=input_mode, injection=injection, queries_per_record=4)
    if variant == "H4":
        return HistoryConfig(variant="H4", input_mode=input_mode, injection=injection, selection="recent8")
    if variant == "H5":
        return HistoryConfig(variant="H5", input_mode=input_mode, injection=injection, mask_numeric=True)
    raise ValueError(f"unknown history variant: {variant}")


class HistoryResidualFusion(nnx.Module):
    """Cross-attend the unchanged current prefix to compressed history behind a zero gate."""

    def __init__(self, *, input_dim: int, width: int, heads: int, rngs: nnx.Rngs):
        self.width = width
        self.heads = heads
        self.q_proj = nnx.Linear(input_dim, width, rngs=rngs)
        self.k_proj = nnx.Linear(input_dim, width, rngs=rngs)
        self.v_proj = nnx.Linear(input_dim, width, rngs=rngs)
        self.output_proj = nnx.Linear(width, input_dim, rngs=rngs)
        self.gate = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def attention_residual(
        self,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        memory_tokens: jax.Array,
        memory_mask: jax.Array,
    ) -> jax.Array:
        batch, prefix_length, _ = prefix_tokens.shape
        memory_length = memory_tokens.shape[1]
        head_dim = self.width // self.heads

        q = self.q_proj(prefix_tokens).reshape(batch, prefix_length, self.heads, head_dim)
        k = self.k_proj(memory_tokens).reshape(batch, memory_length, self.heads, head_dim)
        v = self.v_proj(memory_tokens).reshape(batch, memory_length, self.heads, head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits = logits * (head_dim**-0.5)
        mask = jnp.asarray(memory_mask, dtype=jnp.bool_)[:, None, None, :]
        probabilities = jax.nn.softmax(jnp.where(mask, logits, jnp.finfo(jnp.float32).min), axis=-1)
        probabilities = jnp.where(mask, probabilities, 0.0)
        probabilities = probabilities / jnp.maximum(jnp.sum(probabilities, axis=-1, keepdims=True), 1.0)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probabilities.astype(v.dtype), v)
        residual = self.output_proj(attended.reshape(batch, prefix_length, self.width))

        has_history = jnp.any(memory_mask, axis=-1, keepdims=True)
        output_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_) & has_history
        return jnp.where(output_mask[..., None], residual, 0.0)

    def __call__(
        self,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        memory_tokens: jax.Array,
        memory_mask: jax.Array,
    ) -> jax.Array:
        residual = self.attention_residual(prefix_tokens, prefix_mask, memory_tokens, memory_mask)
        gated_residual = (jnp.tanh(self.gate.value) * residual).astype(prefix_tokens.dtype)
        return prefix_tokens + gated_residual


class HistoryCompressor(nnx.Module):
    """One dense cross-attention block; H1/H2 differ only in the boolean mask."""

    def __init__(
        self,
        config: HistoryConfig,
        *,
        visual_dim: int,
        numeric_dim: int,
        output_dim: int,
        num_cameras: int,
        rngs: nnx.Rngs,
    ):
        self.config = config
        width = config.width
        self.visual_proj = nnx.Linear(visual_dim, width, rngs=rngs)
        self.numeric_proj = nnx.Linear(numeric_dim, width, rngs=rngs)
        self.age_proj = nnx.Linear(1, width, rngs=rngs)
        self.q_proj = nnx.Linear(width, width, rngs=rngs)
        self.k_proj = nnx.Linear(width, width, rngs=rngs)
        self.v_proj = nnx.Linear(width, width, rngs=rngs)
        self.attn_out = nnx.Linear(width, width, rngs=rngs)
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.mlp_in = nnx.Linear(width, config.mlp_hidden, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_hidden, width, rngs=rngs)
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.output_proj = nnx.Linear(width, output_dim, rngs=rngs)
        self.residual_fusion = (
            HistoryResidualFusion(input_dim=output_dim, width=width, heads=config.heads, rngs=rngs)
            if config.injection == "residual"
            else None
        )

        def normal(shape, scale=0.02):
            return jax.random.normal(rngs.params(), shape, dtype=jnp.float32) * scale

        self.queries = nnx.Param(normal((config.num_queries, width)))
        self.camera_embedding = nnx.Param(normal((num_cameras, width)))
        self.frame_embedding = nnx.Param(normal((2, width)))
        self.patch_embedding = nnx.Param(normal((config.max_patches, width)))
        # visual, start state, executed action, state delta
        self.type_embedding = nnx.Param(normal((4, width)))

    def encode_records(self, history, visual_features: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """Return `[B,N,T,512]` sources and `[B,N,T]` validity."""
        visual_tokens = []
        visual_masks = []
        record_mask = jnp.asarray(history.record_mask, dtype=jnp.bool_)
        relative_age = (jnp.asarray(history.query_time)[:, None] - jnp.asarray(history.times)).astype(jnp.float32)
        age_embedding = self.age_proj(jnp.log1p(jnp.maximum(relative_age, 0.0))[..., None])

        for camera_index, name in enumerate(visual_features):
            features = visual_features[name]
            if features.shape[2] != 2:
                raise ValueError(f"history camera {name!r} must contain before/after frames")
            if features.shape[3] > self.config.max_patches:
                raise ValueError(
                    f"history camera {name!r} has {features.shape[3]} patches; "
                    f"configured maximum is {self.config.max_patches}"
                )
            projected = self.visual_proj(features)
            projected = projected + self.camera_embedding[camera_index]
            projected = projected + self.frame_embedding[None, None, :, None, :]
            projected = projected + self.patch_embedding[None, None, None, : features.shape[3], :]
            projected = projected + self.type_embedding[0]
            projected = projected + age_embedding[:, :, None, None, :]
            batch, records, frames, patches, width = projected.shape
            visual_tokens.append(projected.reshape(batch, records, frames * patches, width))
            camera_mask = jnp.asarray(history.image_masks[name], dtype=jnp.bool_)
            camera_mask = camera_mask & record_mask[:, :, None]
            visual_masks.append(jnp.repeat(camera_mask[..., None], patches, axis=-1).reshape(batch, records, -1))

        state_dim_mask = jnp.asarray(history.state_dim_mask, dtype=jnp.bool_)
        action_dim_mask = jnp.asarray(history.action_dim_mask, dtype=jnp.bool_)
        state = jnp.where(state_dim_mask, history.states, 0.0)
        action = jnp.where(action_dim_mask, history.actions, 0.0)
        state_delta = jnp.where(state_dim_mask, history.state_deltas, 0.0)
        numeric_values = (state, action, state_delta)
        numeric_tokens = []
        for type_index, value in enumerate(numeric_values, start=1):
            token = self.numeric_proj(value) + self.type_embedding[type_index] + age_embedding
            numeric_tokens.append(token[:, :, None, :])
        numeric_tokens = jnp.concatenate(numeric_tokens, axis=2)
        numeric_mask = jnp.asarray(history.numeric_masks, dtype=jnp.bool_) & record_mask[:, :, None]
        if self.config.mask_numeric:
            numeric_mask = jnp.zeros_like(numeric_mask)

        sources = jnp.concatenate([*visual_tokens, numeric_tokens], axis=2)
        source_mask = jnp.concatenate([*visual_masks, numeric_mask], axis=2)
        # Invalid contents must be observationally irrelevant.
        sources = jnp.where(source_mask[..., None], sources, 0.0)
        return sources, source_mask

    def __call__(self, history, visual_features: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        sources, source_mask = self.encode_records(history, visual_features)
        batch, records, tokens_per_record, width = sources.shape
        if records != self.config.max_records:
            raise ValueError(f"expected {self.config.max_records} history slots, got {records}")

        flat_sources = sources.reshape(batch, records * tokens_per_record, width)
        flat_mask = source_mask.reshape(batch, records * tokens_per_record)
        query_record = jnp.arange(self.config.num_queries) // self.config.queries_per_record
        query_valid = jnp.take(jnp.asarray(history.record_mask, dtype=jnp.bool_), query_record, axis=1)
        if self.config.joint_attention:
            attention_mask = query_valid[:, :, None] & flat_mask[:, None, :]
        else:
            source_record = jnp.repeat(jnp.arange(records), tokens_per_record)
            local = query_record[:, None] == source_record[None, :]
            attention_mask = query_valid[:, :, None] & flat_mask[:, None, :] & local[None, :, :]

        queries = jnp.broadcast_to(self.queries[None, :, :], (batch, self.config.num_queries, width))
        q = self.q_proj(queries)
        k = self.k_proj(flat_sources)
        v = self.v_proj(flat_sources)
        head_dim = width // self.config.heads
        q = q.reshape(batch, self.config.num_queries, self.config.heads, head_dim)
        k = k.reshape(batch, records * tokens_per_record, self.config.heads, head_dim)
        v = v.reshape(batch, records * tokens_per_record, self.config.heads, head_dim)
        logits = jnp.einsum("bqhd,bshd->bhqs", q, k, preferred_element_type=jnp.float32)
        logits = logits * (head_dim**-0.5)
        mask = attention_mask[:, None, :, :]
        # Softmax is evaluated on a safe row, then re-masked and re-normalized.  A fully
        # masked query therefore produces exact zeros instead of NaN or uniform leakage.
        safe_logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(safe_logits, axis=-1)
        probs = jnp.where(mask, probs, 0.0)
        probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1.0)
        attended = jnp.einsum("bhqs,bshd->bqhd", probs.astype(v.dtype), v).reshape(
            batch, self.config.num_queries, width
        )
        hidden = self.norm1(queries + self.attn_out(attended))
        hidden = self.norm2(hidden + self.mlp_out(jax.nn.gelu(self.mlp_in(hidden))))
        output = self.output_proj(hidden)
        output = jnp.where(query_valid[..., None], output, 0.0)
        return output, query_valid

    def fuse_residual(
        self,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        memory_tokens: jax.Array,
        memory_mask: jax.Array,
    ) -> jax.Array:
        return self.residual_fusion(prefix_tokens, prefix_mask, memory_tokens, memory_mask)
