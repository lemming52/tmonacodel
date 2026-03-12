"""Trackmania Monte Carlo tournament simulator."""

from .aggregation import SimulationResults
from .config import TournamentConfig
from .data_loader import REAL_CUP_RESULTS, REAL_PLAYERS
from .player import derive_skill, resolve_cup_results
from .simulation import run_monte_carlo
from .types import CupResultByName, RealCupResults

__all__ = [
    "TournamentConfig",
    "run_monte_carlo",
    "SimulationResults",
    "REAL_PLAYERS",
    "REAL_CUP_RESULTS",
    "CupResultByName",
    "RealCupResults",
    "resolve_cup_results",
    "derive_skill",
]
