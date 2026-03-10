from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TournamentConfig:
    n_players: int = 128
    n_qualifiers: int = 64
    n_cups: int = 10
    best_of: int = 5
    # Each tuple: (n_rounds, n_eliminated_per_round)
    elimination_stages: tuple[tuple[int, int], ...] = ((24, 2), (15, 1))
    n_simulations: int = 10_000
    random_seed: int | None = 42
    top_n_cutoff: int = 16  # "qualified for next tournament" threshold

    def __post_init__(self) -> None:
        total_eliminated = sum(n * k for n, k in self.elimination_stages)
        if total_eliminated + 1 != self.n_qualifiers:
            raise ValueError(
                f"elimination_stages eliminate {total_eliminated} players but "
                f"n_qualifiers={self.n_qualifiers} requires {self.n_qualifiers - 1} eliminations"
            )
        if self.best_of > self.n_cups:
            raise ValueError(
                f"best_of={self.best_of} cannot exceed n_cups={self.n_cups}"
            )
        if self.n_qualifiers > self.n_players:
            raise ValueError(
                f"n_qualifiers={self.n_qualifiers} cannot exceed n_players={self.n_players}"
            )
        if self.top_n_cutoff > self.n_players:
            raise ValueError(
                f"top_n_cutoff={self.top_n_cutoff} cannot exceed n_players={self.n_players}"
            )
