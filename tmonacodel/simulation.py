from __future__ import annotations
import numpy as np

from .aggregation import SimulationResults
from .config import TournamentConfig
from .player import make_player_pool
from .scoring import build_finish_position_lookup, build_points_table
from .tournament import simulate_tournament


def run_monte_carlo(
    config: TournamentConfig | None = None,
    player_data: list[tuple[str, str]] | None = None,
) -> SimulationResults:
    """
    Run n_simulations full Elite Cup tournaments and return aggregated results.

    Uses a single seeded RNG for full reproducibility.
    If player_data is provided (list of (name, country)), it is forwarded to
    make_player_pool to populate the player pool with real players.
    """
    if config is None:
        config = TournamentConfig()

    rng = np.random.default_rng(config.random_seed)
    players = make_player_pool(config, player_data)
    points_table = build_points_table(config.n_qualifiers)
    finish_pos_lookup = build_finish_position_lookup(config)

    # Preallocate output arrays
    all_tournament_points = np.empty((config.n_simulations, config.n_players), dtype=np.int64)
    all_tournament_ranks = np.empty((config.n_simulations, config.n_players), dtype=np.int64)

    for sim in range(config.n_simulations):
        tournament_points = simulate_tournament(config, points_table, finish_pos_lookup, rng)
        all_tournament_points[sim] = tournament_points

        # Rank: 1 = highest points. Use argsort twice for dense rank.
        # Negate for descending sort; ties get the same rank via scipy-style logic.
        order = np.argsort(-tournament_points, kind="stable")
        ranks = np.empty(config.n_players, dtype=np.int64)
        ranks[order] = np.arange(1, config.n_players + 1)
        all_tournament_ranks[sim] = ranks

    return SimulationResults(
        config=config,
        players=players,
        all_tournament_points=all_tournament_points,
        all_tournament_ranks=all_tournament_ranks,
    )
