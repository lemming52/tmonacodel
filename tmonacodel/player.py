from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import TournamentConfig
from .types import CupResultByName


@dataclass(frozen=True)
class Player:
    player_id: int  # stable 0-based index — key for all numpy array lookups
    name: str
    country: str = ""
    # Extension point: add skill: float = 1.0 here later


def make_player_pool(
    config: TournamentConfig,
    player_data: list[tuple[str, str]] | None = None,
) -> list[Player]:
    """Create a pool of n_players players with sequential IDs.

    If player_data is provided (list of (name, country)), use those for the
    first len(player_data) players; fill the rest with generic placeholders.
    """
    players: list[Player] = []
    if player_data:
        for i, (name, country) in enumerate(player_data[: config.n_players]):
            players.append(Player(player_id=i, name=name, country=country))
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
                Player(player_id=generic_start + j, name=f"Generic_{j:02d}", country=assigned[j])
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
