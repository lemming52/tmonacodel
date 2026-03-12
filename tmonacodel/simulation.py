from __future__ import annotations
import numpy as np

from .aggregation import SimulationResults
from .config import TournamentConfig
from .player import derive_skill, make_player_pool, resolve_cup_results
from .scoring import build_finish_position_lookup, build_points_table
from .tournament import simulate_tournament
from .types import RealCupResults


def run_monte_carlo(
    config: TournamentConfig | None = None,
    player_data: list[tuple[str, str]] | None = None,
    *,
    real_cup_results: RealCupResults | None = None,
) -> SimulationResults:
    """
    Run n_simulations full Elite Cup tournaments and return aggregated results.

    Uses a single seeded RNG for full reproducibility.
    If player_data is provided (list of (name, country)), it is forwarded to
    make_player_pool to populate the player pool with real players.

    If real_cup_results is provided, each non-None entry replaces the
    corresponding simulated cup with fixed historical points. Trailing cups
    beyond len(real_cup_results) are always simulated.
    """
    if config is None:
        config = TournamentConfig()

    rng = np.random.default_rng(config.random_seed)

    # Derive skills from real results before building player pool
    if real_cup_results is not None:
        completed = [e for e in real_cup_results if e is not None]
        skill_map, generic_skill = derive_skill(completed, [])
    else:
        skill_map, generic_skill = None, 1.0

    players = make_player_pool(
        config, player_data,
        skill_map=skill_map,
        generic_skill=generic_skill,
    )

    # Build player_skills array from the player pool
    if skill_map is not None:
        player_skills: np.ndarray | None = np.array([p.skill for p in players], dtype=float)
    else:
        player_skills = None

    points_table = build_points_table(config.n_qualifiers)
    finish_pos_lookup = build_finish_position_lookup(config)

    # Resolve real results once before the sim loop
    if real_cup_results is not None:
        padded = list(real_cup_results) + [None] * (config.n_cups - len(real_cup_results))
        simulate_mask = np.array([e is None for e in padded], dtype=bool)
        fixed_cup_points = np.zeros((config.n_cups, config.n_players), dtype=np.int64)
        for m, entry in enumerate(padded):
            if entry is not None:
                fixed_cup_points[m] = resolve_cup_results(entry, players)
    else:
        fixed_cup_points = simulate_mask = None

    # Preallocate output arrays
    all_tournament_points = np.empty((config.n_simulations, config.n_players), dtype=np.int64)
    all_tournament_ranks = np.empty((config.n_simulations, config.n_players), dtype=np.int64)

    for sim in range(config.n_simulations):
        tournament_points = simulate_tournament(
            config, points_table, finish_pos_lookup, rng,
            fixed_cup_points=fixed_cup_points,
            simulate_mask=simulate_mask,
            player_skills=player_skills,
        )
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
