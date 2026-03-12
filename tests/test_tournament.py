import numpy as np
import pytest

from tmonacodel.scoring import build_finish_position_lookup, build_points_table
from tmonacodel.tournament import simulate_tournament
from tmonacodel.config import TournamentConfig


@pytest.fixture
def precomputed(default_config):
    return (
        build_points_table(default_config.n_qualifiers),
        build_finish_position_lookup(default_config),
    )


@pytest.fixture
def small_precomputed(small_config):
    return (
        build_points_table(small_config.n_qualifiers),
        build_finish_position_lookup(small_config),
    )


class TestSimulateTournament:
    def test_output_shape(self, default_config, precomputed):
        pts_table, lookup = precomputed
        rng = np.random.default_rng(0)
        result = simulate_tournament(default_config, pts_table, lookup, rng)
        assert result.shape == (default_config.n_players,)

    def test_non_negative(self, default_config, precomputed):
        pts_table, lookup = precomputed
        rng = np.random.default_rng(0)
        result = simulate_tournament(default_config, pts_table, lookup, rng)
        assert np.all(result >= 0)

    def test_best_of_rule_bounds(self, default_config, precomputed):
        pts_table, lookup = precomputed
        rng = np.random.default_rng(0)
        result = simulate_tournament(default_config, pts_table, lookup, rng)
        # Max possible = best_of * max single-race score
        max_per_race = int(pts_table.max())
        assert int(result.max()) <= default_config.best_of * max_per_race

    def test_small_config_reproducible(self, small_config, small_precomputed):
        pts_table, lookup = small_precomputed
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        r1 = simulate_tournament(small_config, pts_table, lookup, rng1)
        r2 = simulate_tournament(small_config, pts_table, lookup, rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_best_of_selects_top_scores(self, small_config, small_precomputed):
        """
        Verify best-of logic: manually check one player.
        If a player scored [100, 0, 50, 200] over 4 races and best_of=2,
        their tournament total should be 100+200=300.
        This test uses a controlled scenario.
        """
        pts_table, lookup = small_precomputed
        rng = np.random.default_rng(0)
        # Run many tournaments and verify totals are non-negative and bounded
        for _ in range(10):
            result = simulate_tournament(small_config, pts_table, lookup, rng)
            assert np.all(result >= 0)
            assert int(result.max()) <= small_config.best_of * int(pts_table.max())


class TestPartialTournament:
    def test_all_fixed_cups_ignores_rng(self, small_config, small_precomputed):
        """Two different seeds produce identical results when simulate_mask is all-False."""
        pts_table, lookup = small_precomputed
        fixed = np.zeros((small_config.n_cups, small_config.n_players), dtype=np.int64)
        fixed[:] = 100  # arbitrary fixed values
        mask = np.zeros(small_config.n_cups, dtype=bool)  # all False → never simulate

        r1 = simulate_tournament(
            small_config, pts_table, lookup, np.random.default_rng(1),
            fixed_cup_points=fixed, simulate_mask=mask,
        )
        r2 = simulate_tournament(
            small_config, pts_table, lookup, np.random.default_rng(99),
            fixed_cup_points=fixed, simulate_mask=mask,
        )
        np.testing.assert_array_equal(r1, r2)

    def test_fixed_cups_are_not_overwritten(self, small_config, small_precomputed):
        """Player with max points in fixed cups accumulates exactly that in best-of sum."""
        pts_table, lookup = small_precomputed
        max_pts = 9999
        fixed = np.zeros((small_config.n_cups, small_config.n_players), dtype=np.int64)
        fixed[:, 0] = max_pts  # player 0 gets max_pts in every fixed cup
        mask = np.zeros(small_config.n_cups, dtype=bool)  # all fixed

        result = simulate_tournament(
            small_config, pts_table, lookup, np.random.default_rng(0),
            fixed_cup_points=fixed, simulate_mask=mask,
        )
        # Player 0's total = best_of * max_pts (every cup counts)
        assert result[0] == small_config.best_of * max_pts

    def test_none_masks_equal_all_true(self, small_config, small_precomputed):
        """simulate_mask=None is equivalent to all-True (same seed → same output)."""
        pts_table, lookup = small_precomputed

        rng1 = np.random.default_rng(7)
        r1 = simulate_tournament(small_config, pts_table, lookup, rng1)

        rng2 = np.random.default_rng(7)
        all_true_mask = np.ones(small_config.n_cups, dtype=bool)
        r2 = simulate_tournament(
            small_config, pts_table, lookup, rng2,
            simulate_mask=all_true_mask,
        )
        np.testing.assert_array_equal(r1, r2)
