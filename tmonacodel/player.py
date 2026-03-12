from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import TournamentConfig
from .types import CupResultByName


@dataclass(frozen=True)
class Player:
    player_id: int  # stable 0-based index — key for all numpy array lookups
    name: str
    country: str = ""
    skill: float = 1.0  # relative weight; 1.0 = neutral


def derive_skill(
    real_cup_results: list[CupResultByName],
    players: list[Player],
) -> tuple[dict[str, float], float]:
    """Derive per-player skill from historical cup results.

    For each named real player, computes mean(points) / 1000.0 across all
    cups in which they appear. Clipped to [0.01, 1.0].

    Returns (skill_map, generic_skill) where generic_skill is the 25th
    percentile of derived skill values (used for unnamed fill players).
    """
    if not real_cup_results:
        return {}, 1.0

    # Accumulate points per player name
    point_sums: dict[str, float] = {}
    appearance_counts: dict[str, int] = {}
    for cup_result in real_cup_results:
        for name, pts in cup_result.items():
            point_sums[name] = point_sums.get(name, 0.0) + pts
            appearance_counts[name] = appearance_counts.get(name, 0) + 1

    skill_map: dict[str, float] = {}
    for name, total in point_sums.items():
        mean_pts = total / appearance_counts[name]
        raw_skill = mean_pts / 1000.0
        skill_map[name] = float(np.clip(raw_skill, 0.01, 1.0))

    skill_values = np.array(list(skill_map.values()), dtype=float)
    generic_skill = float(np.percentile(skill_values, 25))

    return skill_map, generic_skill


def make_player_pool(
    config: TournamentConfig,
    player_data: list[tuple[str, str]] | None = None,
    *,
    skill_map: dict[str, float] | None = None,
    generic_skill: float = 1.0,
) -> list[Player]:
    """Create a pool of n_players players with sequential IDs.

    If player_data is provided (list of (name, country)), use those for the
    first len(player_data) players; fill the rest with generic placeholders.
    """
    players: list[Player] = []
    if player_data:
        for i, (name, country) in enumerate(player_data[: config.n_players]):
            skill = skill_map.get(name, generic_skill) if skill_map else 1.0
            players.append(Player(player_id=i, name=name, country=country, skill=skill))
    generic_start = len(players)
    n_generics = config.n_players - generic_start
    if n_generics > 0:
        if player_data:
            nations = [c for _, c in player_data if c]
            nation_names, counts = zip(*Counter(nations).items())
            weights = np.array(counts, dtype=float) / sum(counts)
            rng = np.random.default_rng(config.random_seed)
            assigned: list[str] = list(rng.choice(nation_names, size=n_generics, p=weights))
        else:
            assigned = [""] * n_generics
        for j in range(n_generics):
            players.append(
                Player(
                    player_id=generic_start + j,
                    name=f"Generic_{j:02d}",
                    country=assigned[j],
                    skill=generic_skill,
                )
            )
    return players


def resolve_cup_results(
    results_by_name: CupResultByName,
    players: list[Player],
) -> np.ndarray:
    """Convert a name-keyed cup result dict into a player-id-indexed array.

    Returns an ndarray of shape (n_players,) with dtype int64.
    Unknown names are silently dropped (contribute 0 points).
    """
    name_to_id = {p.name: p.player_id for p in players}
    out = np.zeros(len(players), dtype=np.int64)
    for name, pts in results_by_name.items():
        pid = name_to_id.get(name)
        if pid is not None:
            out[pid] = pts
    return out
