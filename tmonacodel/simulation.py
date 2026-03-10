from __future__ import annotations
import numpy as np

from .aggregation import SimulationResults
from .config import TournamentConfig
from .racer import make_racer_pool
from .scoring import build_finish_position_lookup, build_points_table
from .season import simulate_season


def run_monte_carlo(
    config: TournamentConfig | None = None,
    player_data: list[tuple[str, str]] | None = None,
) -> SimulationResults:
    """
    Run n_simulations full seasons and return aggregated results.

    Uses a single seeded RNG for full reproducibility.
    If player_data is provided (list of (name, country)), it is forwarded to
    make_racer_pool to populate the racer pool with real players.
    """
    if config is None:
        config = TournamentConfig()

    rng = np.random.default_rng(config.random_seed)
    racers = make_racer_pool(config, player_data)
    points_table = build_points_table(config.n_qualifiers)
    finish_pos_lookup = build_finish_position_lookup(config)

    # Preallocate output arrays
    all_season_points = np.empty((config.n_simulations, config.n_players), dtype=np.int64)
    all_season_ranks = np.empty((config.n_simulations, config.n_players), dtype=np.int64)

    for sim in range(config.n_simulations):
        season_points = simulate_season(config, points_table, finish_pos_lookup, rng)
        all_season_points[sim] = season_points

        # Rank: 1 = highest points. Use argsort twice for dense rank.
        # Negate for descending sort; ties get the same rank via scipy-style logic.
        order = np.argsort(-season_points, kind="stable")
        ranks = np.empty(config.n_players, dtype=np.int64)
        ranks[order] = np.arange(1, config.n_players + 1)
        all_season_ranks[sim] = ranks

    return SimulationResults(
        config=config,
        racers=racers,
        all_season_points=all_season_points,
        all_season_ranks=all_season_ranks,
    )
