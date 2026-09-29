import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.history as _history
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Experimental executed-history interface. None is the exact upstream H0 path.
    history: _history.HistoryConfig | None = None
    # H0--H5 freeze the shared current/history SigLIP while updating the VLM and
    # action expert.  Kept explicit so unrelated upstream configs are unchanged.
    freeze_vision: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        base_rng, history_rng = jax.random.split(rng)
        return Pi0(self, rngs=nnx.Rngs(base_rng), history_rngs=nnx.Rngs(history_rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            history_spec = None
            if self.history is not None:
                n = self.history.max_records
                frame_mask_spec = jax.ShapeDtypeStruct([batch_size, n, 2], jnp.bool_)
                history_images = {}
                history_features = {}
                if self.history.input_mode == "raw":
                    history_images = {
                        key: jax.ShapeDtypeStruct(
                            [batch_size, n, 2, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                        )
                        for key in _model.IMAGE_KEYS[:2]
                    }
                else:
                    visual_dim = _gemma.get_config(self.paligemma_variant).width
                    history_features = {
                        key: jax.ShapeDtypeStruct(
                            [batch_size, n, 2, self.history.max_patches, visual_dim], jnp.float32
                        )
                        for key in _model.IMAGE_KEYS[:2]
                    }
                history_spec = _model.HistoryObservation(
                    images=history_images,
                    image_features=history_features,
                    image_masks={key: frame_mask_spec for key in _model.IMAGE_KEYS[:2]},
                    record_mask=jax.ShapeDtypeStruct([batch_size, n], jnp.bool_),
                    states=jax.ShapeDtypeStruct([batch_size, n, self.action_dim], jnp.float32),
                    actions=jax.ShapeDtypeStruct([batch_size, n, self.action_dim], jnp.float32),
                    state_deltas=jax.ShapeDtypeStruct([batch_size, n, self.action_dim], jnp.float32),
                    numeric_masks=jax.ShapeDtypeStruct([batch_size, n, 3], jnp.bool_),
                    state_dim_mask=jax.ShapeDtypeStruct([batch_size, n, self.action_dim], jnp.bool_),
                    action_dim_mask=jax.ShapeDtypeStruct([batch_size, n, self.action_dim], jnp.bool_),
                    times=jax.ShapeDtypeStruct([batch_size, n], jnp.int32),
                    query_time=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                )
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                history=history_spec,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if self.freeze_vision:
            filters.append(nnx_utils.PathRegex(".*PaliGemma/img.*"))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
