from __future__ import annotations
import numpy as np


def simulate_elimination(
    n_qualifiers: int,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one match's elimination stage.

    Returns finish_positions: ndarray shape (n_qualifiers,)
      finish_positions[i] = finish position of qualifier i (1-based, lower is better)

    Hot path: one rng.permutation + one fancy-index scatter.

    Extension point: replace rng.permutation with weighted rng.choice for skill.
    """
    # perm[i] = elimination order index for qualifier i
    # perm is a random permutation of [0, n_qualifiers)
    # perm[i] = 0 means qualifier i is eliminated first (worst finish)
    # perm[i] = n_qualifiers-1 means qualifier i is last standing (1st place)
    perm = rng.permutation(n_qualifiers)

    finish_positions = np.empty(n_qualifiers, dtype=np.int64)
    finish_positions[perm] = finish_pos_lookup

    return finish_positions
