from __future__ import annotations
import numpy as np

from .config import TournamentConfig
from .race import simulate_race
from .scoring import build_points_table, build_finish_position_lookup


def qualify_players(
    n_players: int,
    n_qualifiers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly select n_qualifiers players from n_players.

    Returns qualifier_ids: ndarray shape (n_qualifiers,) of player indices.

    Extension point: add weights based on player skill for biased qualification.
    """
    return rng.choice(n_players, size=n_qualifiers, replace=False)


def simulate_cup(
    config: TournamentConfig,
    points_table: np.ndarray,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one cup (Cup of the Week): qualification then race.

    Returns race_points: ndarray shape (n_players,)
      race_points[i] = points earned by player i (0 if did not qualify).
    """
    race_points = np.zeros(config.n_players, dtype=np.int64)

    qualifier_ids = qualify_players(config.n_players, config.n_qualifiers, rng)
    finish_positions = simulate_race(config.n_qualifiers, finish_pos_lookup, rng)

    # finish_positions[j] = finish position of qualifier_ids[j]
    # points_table index: finish position (1-based), 0 = did not qualify
    race_points[qualifier_ids] = points_table[finish_positions]

    return race_points
