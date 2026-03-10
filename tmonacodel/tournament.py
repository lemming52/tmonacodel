from __future__ import annotations
import numpy as np

from .config import TournamentConfig
from .cup import simulate_cup


def simulate_tournament(
    config: TournamentConfig,
    points_table: np.ndarray,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one full Elite Cup tournament (n_cups cups).

    Returns tournament_points: ndarray shape (n_players,)
      Sum of player's best `config.best_of` cup scores.
    """
    # cup_points shape: (n_cups, n_players)
    cup_points = np.empty((config.n_cups, config.n_players), dtype=np.int64)

    for m in range(config.n_cups):
        cup_points[m] = simulate_cup(config, points_table, finish_pos_lookup, rng)

    # Best-N-of-M rule: sum of top `best_of` scores per player
    # Sort descending along cup axis (axis=0), independently per player column
    sorted_desc = np.sort(cup_points, axis=0)[::-1]
    tournament_points = sorted_desc[: config.best_of].sum(axis=0)

    return tournament_points
