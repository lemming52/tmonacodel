from __future__ import annotations
import numpy as np

from .config import TournamentConfig
from .race import simulate_race
from .scoring import build_points_table, build_finish_position_lookup


def qualify_players(
    n_players: int,
    n_qualifiers: int,
    rng: np.random.Generator,
    skill_weights: np.ndarray | None = None,  # shape (n_players,), unnormalised
) -> np.ndarray:
    """
    Randomly select n_qualifiers players from n_players.

    Returns qualifier_ids: ndarray shape (n_qualifiers,) of player indices.

    When skill_weights is provided, qualification probability is proportional to skill.
    """
    if skill_weights is not None:
        weights = skill_weights / skill_weights.sum()
        return rng.choice(n_players, size=n_qualifiers, replace=False, p=weights)
    return rng.choice(n_players, size=n_qualifiers, replace=False)


def simulate_cup(
    config: TournamentConfig,
    points_table: np.ndarray,
    finish_pos_lookup: np.ndarray,
    rng: np.random.Generator,
    player_skills: np.ndarray | None = None,  # shape (n_players,)
) -> np.ndarray:
    """
    Simulate one cup (Cup of the Week): qualification then race.

    Returns race_points: ndarray shape (n_players,)
      race_points[i] = points earned by player i (0 if did not qualify).
    """
    race_points = np.zeros(config.n_players, dtype=np.int64)

    qualifier_ids = qualify_players(config.n_players, config.n_qualifiers, rng, skill_weights=player_skills)
    qualifier_skills = player_skills[qualifier_ids] if player_skills is not None else None
    finish_positions = simulate_race(config.n_qualifiers, finish_pos_lookup, rng, qualifier_skills=qualifier_skills)

    # finish_positions[j] = finish position of qualifier_ids[j]
    # points_table index: finish position (1-based), 0 = did not qualify
    race_points[qualifier_ids] = points_table[finish_positions]

    return race_points
