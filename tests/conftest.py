import pytest
from tmonacodel.config import TournamentConfig


@pytest.fixture
def default_config():
    return TournamentConfig()


@pytest.fixture
def small_config():
    """A smaller config for faster tests."""
    return TournamentConfig(
        n_players=20,
        n_qualifiers=8,
        n_cups=4,
        best_of=2,
        elimination_stages=((3, 2), (1, 1)),  # 6 + 1 = 7 elim + 1 survivor = 8
        n_simulations=100,
        random_seed=0,
        top_n_cutoff=4,
    )
