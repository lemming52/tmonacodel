from __future__ import annotations
import numpy as np

from .config import TournamentConfig
from .match import simulate_match


def simulate_season(
    config: TournamentConfig,
    points_table: np.ndarray,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one full season (n_matches matches).

    Returns season_points: ndarray shape (n_players,)
      Sum of player's best `config.best_of` match scores.
    """
    # match_points shape: (n_matches, n_players)
    match_points = np.empty((config.n_matches, config.n_players), dtype=np.int64)

    for m in range(config.n_matches):
        match_points[m] = simulate_match(config, points_table, finish_pos_lookup, rng)

    # Best-N-of-M rule: sum of top `best_of` scores per player
    # Sort descending along match axis (axis=0), independently per player column
    sorted_desc = np.sort(match_points, axis=0)[::-1]
    season_points = sorted_desc[: config.best_of].sum(axis=0)

    return season_points
