"""Trackmania Monte Carlo tournament simulator."""

from .aggregation import SimulationResults
from .config import TournamentConfig
from .data_loader import REAL_PLAYERS
from .player import resolve_cup_results
from .simulation import run_monte_carlo
from .types import CupResultByName, RealCupResults

__all__ = [
    "TournamentConfig",
    "run_monte_carlo",
    "SimulationResults",
    "REAL_PLAYERS",
    "CupResultByName",
    "RealCupResults",
    "resolve_cup_results",
]
