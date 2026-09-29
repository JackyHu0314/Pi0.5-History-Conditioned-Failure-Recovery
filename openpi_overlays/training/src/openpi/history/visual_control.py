"""Visual-content controls for executed-history inputs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def broadcast_current_features(
    current_features: Mapping[str, np.ndarray],
    record_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Broadcast the query observation into valid history slots.

    The output has the same ``[record, before/after, patch, feature]`` layout as
    cached executed history. Invalid records remain exactly zero. This function
    never indexes a temporal source, which makes access to query+1 impossible.
    """
    mask = np.asarray(record_mask, dtype=np.bool_)
    if mask.shape != (8,):
        raise ValueError(f"record_mask must have shape (8,), got {mask.shape}")
    outputs: dict[str, np.ndarray] = {}
    for camera, feature in current_features.items():
        value = np.asarray(feature)
        result = np.zeros((8, 2, *value.shape), dtype=value.dtype)
        result[mask, 0] = value
        result[mask, 1] = value
        outputs[camera] = result
    return outputs
