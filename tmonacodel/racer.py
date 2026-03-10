from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import TournamentConfig


@dataclass(frozen=True)
class Racer:
    racer_id: int  # stable 0-based index — key for all numpy array lookups
    name: str
    country: str = ""
    # Extension point: add skill: float = 1.0 here later


def make_racer_pool(
    config: TournamentConfig,
    player_data: list[tuple[str, str]] | None = None,
) -> list[Racer]:
    """Create a pool of n_players racers with sequential IDs.

    If player_data is provided (list of (name, country)), use those for the
    first len(player_data) racers; fill the rest with generic placeholders.
    """
    racers: list[Racer] = []
    if player_data:
        for i, (name, country) in enumerate(player_data[: config.n_players]):
            racers.append(Racer(racer_id=i, name=name, country=country))
    generic_start = len(racers)
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
            racers.append(
                Racer(racer_id=generic_start + j, name=f"Generic_{j:02d}", country=assigned[j])
            )
    return racers
