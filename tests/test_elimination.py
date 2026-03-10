import numpy as np
import pytest

from tmonacodel.config import TournamentConfig
from tmonacodel.elimination import simulate_elimination
from tmonacodel.scoring import build_finish_position_lookup


@pytest.fixture
def default_lookup(default_config):
    return build_finish_position_lookup(default_config)


@pytest.fixture
def small_lookup(small_config):
    return build_finish_position_lookup(small_config)


class TestSimulateElimination:
    def test_output_shape(self, default_config, default_lookup):
        rng = np.random.default_rng(0)
        result = simulate_elimination(default_config.n_qualifiers, default_lookup, rng)
        assert result.shape == (default_config.n_qualifiers,)

    def test_one_winner(self, default_config, default_lookup):
        rng = np.random.default_rng(0)
        result = simulate_elimination(default_config.n_qualifiers, default_lookup, rng)
        assert np.sum(result == 1) == 1

    def test_positions_in_valid_range(self, default_config, default_lookup):
        rng = np.random.default_rng(0)
        result = simulate_elimination(default_config.n_qualifiers, default_lookup, rng)
        assert result.min() >= 1
        assert result.max() <= default_config.n_qualifiers

    def test_all_players_get_position(self, default_config, default_lookup):
        rng = np.random.default_rng(0)
        result = simulate_elimination(default_config.n_qualifiers, default_lookup, rng)
        assert len(result) == default_config.n_qualifiers
        # No zeros — all qualifiers get a finish position
        assert np.all(result > 0)

    def test_reproducible_with_same_seed(self, default_config, default_lookup):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = simulate_elimination(default_config.n_qualifiers, default_lookup, rng1)
        r2 = simulate_elimination(default_config.n_qualifiers, default_lookup, rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self, default_config, default_lookup):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        r1 = simulate_elimination(default_config.n_qualifiers, default_lookup, rng1)
        r2 = simulate_elimination(default_config.n_qualifiers, default_lookup, rng2)
        assert not np.array_equal(r1, r2)

    def test_small_config(self, small_config, small_lookup):
        rng = np.random.default_rng(0)
        result = simulate_elimination(small_config.n_qualifiers, small_lookup, rng)
        assert result.shape == (small_config.n_qualifiers,)
        assert np.sum(result == 1) == 1
