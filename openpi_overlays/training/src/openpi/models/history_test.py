from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import history as _history


def _make_fusion() -> _history.HistoryResidualFusion:
    return _history.HistoryResidualFusion(input_dim=16, width=8, heads=2, rngs=nnx.Rngs(0))


def _inputs():
    prefix = jax.random.normal(jax.random.key(1), (2, 5, 16)).astype(jnp.bfloat16)
    prefix_mask = jnp.array([[True, True, True, True, False], [True, True, True, False, False]])
    memory = jax.random.normal(jax.random.key(2), (2, 4, 16))
    memory_mask = jnp.array([[True, True, False, False], [True, True, True, False]])
    return prefix, prefix_mask, memory, memory_mask


def _gradient_norm(state: nnx.State) -> jax.Array:
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(state))


def _fusion_weight_gradient_norm(grads: nnx.State) -> jax.Array:
    return sum(_gradient_norm(grads[name]) for name in ("q_proj", "k_proj", "v_proj", "output_proj"))


def _make_compressor_inputs():
    compressor = _history.HistoryCompressor(
        _history.HistoryConfig(
            variant="H2",
            injection="residual",
            max_records=2,
            width=8,
            heads=2,
            mlp_hidden=16,
            max_patches=2,
        ),
        visual_dim=16,
        numeric_dim=3,
        output_dim=16,
        num_cameras=1,
        rngs=nnx.Rngs(0),
    )
    history = SimpleNamespace(
        record_mask=jnp.array([[True, True], [True, False]]),
        image_masks={"cam": jnp.ones((2, 2, 2), dtype=jnp.bool_)},
        states=jax.random.normal(jax.random.key(3), (2, 2, 3)),
        actions=jax.random.normal(jax.random.key(4), (2, 2, 3)),
        state_deltas=jax.random.normal(jax.random.key(5), (2, 2, 3)),
        numeric_masks=jnp.ones((2, 2, 3), dtype=jnp.bool_),
        state_dim_mask=jnp.ones((2, 2, 3), dtype=jnp.bool_),
        action_dim_mask=jnp.ones((2, 2, 3), dtype=jnp.bool_),
        times=jnp.array([[0, 1], [0, 0]], dtype=jnp.int32),
        query_time=jnp.array([2, 1], dtype=jnp.int32),
    )
    visual_features = {"cam": jax.random.normal(jax.random.key(6), (2, 2, 2, 2, 16))}
    return compressor, history, visual_features


def _compressor_gradient_norm(grads: nnx.State) -> jax.Array:
    return sum(_gradient_norm(subtree) for name, subtree in grads.items() if name != "residual_fusion")


def test_residual_fusion_preserves_prefix_at_zero_gate_and_without_history():
    fusion = _make_fusion()
    prefix, prefix_mask, memory, memory_mask = _inputs()

    zero_gate_output = fusion(prefix, prefix_mask, memory, memory_mask)
    assert zero_gate_output.shape == prefix.shape
    assert zero_gate_output.dtype == prefix.dtype
    np.testing.assert_array_equal(zero_gate_output, prefix)

    fusion.gate.value = jnp.array(1.0)
    empty_history_mask = jnp.zeros_like(memory_mask)
    empty_residual = fusion.attention_residual(prefix, prefix_mask, memory, empty_history_mask)
    np.testing.assert_array_equal(empty_residual, jnp.zeros_like(empty_residual))
    empty_history_output = fusion(prefix, prefix_mask, memory, empty_history_mask)
    np.testing.assert_array_equal(empty_history_output, prefix)


def test_zero_gate_opens_before_fusion_weights_receive_gradients():
    compressor, history, visual_features = _make_compressor_inputs()
    prefix, prefix_mask, _, _ = _inputs()
    memory, memory_mask = compressor(history, visual_features)
    target_direction = jax.lax.stop_gradient(
        compressor.residual_fusion.attention_residual(prefix, prefix_mask, memory, memory_mask)
    )

    def loss_fn(module):
        memory_tokens, current_memory_mask = module(history, visual_features)
        output = module.fuse_residual(prefix, prefix_mask, memory_tokens, current_memory_mask)
        return jnp.sum(output * target_direction)

    _, first_grads = nnx.value_and_grad(loss_fn)(compressor)
    assert jnp.abs(first_grads.residual_fusion.gate.value) > 0
    np.testing.assert_array_equal(_fusion_weight_gradient_norm(first_grads.residual_fusion), 0.0)
    np.testing.assert_array_equal(_compressor_gradient_norm(first_grads), 0.0)

    compressor.residual_fusion.gate.value -= 0.1 * first_grads.residual_fusion.gate.value
    _, second_grads = nnx.value_and_grad(loss_fn)(compressor)
    assert _fusion_weight_gradient_norm(second_grads.residual_fusion) > 0
    assert _compressor_gradient_norm(second_grads) > 0
