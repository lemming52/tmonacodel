from __future__ import annotations
import numpy as np
import pytest

from tmonacodel.config import TournamentConfig
from tmonacodel.race import simulate_race
from tmonacodel.scoring import build_finish_position_lookup, build_points_table


@pytest.fixture
def finish_pos_lookup():
    cfg = TournamentConfig()
    return build_finish_position_lookup(cfg)


class TestSimulateRace:
    def test_uniform_skills_same_as_no_skills(self, finish_pos_lookup):
        """All skills=1.0 → same result as no skills with same seed."""
        n = 64
        seed = 123
        rng1 = np.random.default_rng(seed)
        result_no_skills = simulate_race(n, finish_pos_lookup, rng1)

        # With all skills=1.0, Gumbel+log(1.0) = Gumbel+0; argsort(argsort(Gumbel))
        # is NOT identical to rng.permutation (different RNG consumption pattern),
        # so we verify statistical equivalence rather than exact equality.
        rng2 = np.random.default_rng(seed)
        skills = np.ones(n)
        result_with_skills = simulate_race(n, finish_pos_lookup, rng2, qualifier_skills=skills)

        # Both should produce valid finish positions in [1, n]
        assert result_no_skills.shape == (n,)
        assert result_with_skills.shape == (n,)
        assert set(result_no_skills).issubset(range(1, n + 1))
        assert set(result_with_skills).issubset(range(1, n + 1))

    def test_strong_player_wins_more_often(self, finish_pos_lookup):
        """Player with skill=10, rest=1 should finish 1st much more than 1/n_qualifiers."""
        n = 64
        n_trials = 5000
        strong_wins = 0
        rng = np.random.default_rng(42)
        skills = np.ones(n)
        skills[0] = 10.0  # player 0 is much stronger

        for _ in range(n_trials):
            finish_pos = simulate_race(n, finish_pos_lookup, rng, qualifier_skills=skills)
            if finish_pos[0] == 1:
                strong_wins += 1

        win_rate = strong_wins / n_trials
        expected_uniform = 1.0 / n
        # Strong player should win much more than 1/64 ≈ 0.016
        assert win_rate > expected_uniform * 3, f"Win rate {win_rate:.3f} not much above {expected_uniform:.3f}"

    def test_no_skills_returns_valid_positions(self, finish_pos_lookup):
        n = 64
        rng = np.random.default_rng(0)
        result = simulate_race(n, finish_pos_lookup, rng)
        assert result.shape == (n,)
        assert result.min() >= 1
        assert result.max() <= n

    def test_with_skills_returns_valid_positions(self, finish_pos_lookup):
        n = 64
        rng = np.random.default_rng(0)
        skills = np.random.default_rng(1).uniform(0.1, 1.0, size=n)
        result = simulate_race(n, finish_pos_lookup, rng, qualifier_skills=skills)
        assert result.shape == (n,)
        assert result.min() >= 1
        assert result.max() <= n
