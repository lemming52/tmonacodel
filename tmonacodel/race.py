from __future__ import annotations
import numpy as np


def simulate_race(
    n_qualifiers: int,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
    qualifier_skills: np.ndarray | None = None,  # shape (n_qualifiers,)
) -> np.ndarray:
    """
    Simulate the race stage of a cup: 39 rounds of elimination for n_qualifiers players.

    Returns finish_positions: ndarray shape (n_qualifiers,)
      finish_positions[i] = finish position of qualifier i (1-based, lower is better)

    When qualifier_skills is provided, uses Plackett-Luce ordering via Gumbel noise:
      scores = Gumbel() + log(skill)  — higher skill → higher expected score → better finish.
    When all skills = 1.0, log(skill) = 0 and ordering is equivalent to uniform permutation.
    """
    if qualifier_skills is not None:
        scores = rng.gumbel(size=n_qualifiers) + np.log(qualifier_skills)
        # argsort gives player indices sorted by score ascending:
        #   perm[0] = player with worst score, perm[n-1] = player with best score
        # This matches the original rng.permutation semantics used by finish_positions[perm] = finish_pos_lookup
        perm = np.argsort(scores)
    else:
        # perm[i] = 0 means qualifier i is eliminated first (worst finish)
        # perm[i] = n_qualifiers-1 means qualifier i is last standing (1st place)
        perm = rng.permutation(n_qualifiers)

    finish_positions = np.empty(n_qualifiers, dtype=np.int64)
    finish_positions[perm] = finish_pos_lookup

    return finish_positions
