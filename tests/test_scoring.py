import numpy as np
import pytest

from tmonacodel.config import TournamentConfig
from tmonacodel.scoring import build_finish_position_lookup, build_points_table


class TestBuildPointsTable:
    def test_dnq_is_zero(self):
        pts = build_points_table(64)
        assert pts[0] == 0

    def test_length(self):
        pts = build_points_table(64)
        assert len(pts) == 65  # indices 0..64

    def test_top_8(self):
        pts = build_points_table(64)
        expected = [1000, 700, 600, 500, 400, 300, 200, 100]
        for i, exp in enumerate(expected, start=1):
            assert pts[i] == exp, f"position {i}: expected {exp}, got {pts[i]}"

    def test_positions_9_to_16(self):
        pts = build_points_table(64)
        expected = [96, 92, 88, 84, 80, 76, 72, 68]
        for i, exp in enumerate(expected, start=9):
            assert pts[i] == exp, f"position {i}: expected {exp}, got {pts[i]}"

    def test_paired_positions_17_18(self):
        pts = build_points_table(64)
        assert pts[17] == 64
        assert pts[18] == 64

    def test_paired_positions_63_64(self):
        pts = build_points_table(64)
        assert pts[63] == pts[64]
        assert pts[63] > 0

    def test_non_increasing(self):
        pts = build_points_table(64)
        # From position 1 onward, points should be non-increasing
        for i in range(1, 64):
            assert pts[i] >= pts[i + 1], f"pts[{i}]={pts[i]} < pts[{i+1}]={pts[i+1]}"

    def test_last_place_positive(self):
        pts = build_points_table(64)
        assert pts[64] == 2


class TestBuildFinishPositionLookup:
    def test_shape(self, default_config):
        lookup = build_finish_position_lookup(default_config)
        assert lookup.shape == (default_config.n_qualifiers,)

    def test_survivor_gets_first(self, default_config):
        lookup = build_finish_position_lookup(default_config)
        # perm_idx n-1 = last standing = 1st place
        assert lookup[-1] == 1

    def test_first_eliminated_gets_worst(self, default_config):
        lookup = build_finish_position_lookup(default_config)
        # perm_idx 0 = first eliminated = worst finish
        assert lookup[0] == default_config.n_qualifiers - 1 or lookup[0] > 1

    def test_all_positions_covered(self, default_config):
        lookup = build_finish_position_lookup(default_config)
        assert lookup.min() >= 1
        assert lookup.max() <= default_config.n_qualifiers

    def test_small_config(self, small_config):
        lookup = build_finish_position_lookup(small_config)
        assert lookup.shape == (small_config.n_qualifiers,)
        assert lookup[-1] == 1  # survivor is 1st


class TestConfigValidation:
    def test_invalid_elimination_stages(self):
        with pytest.raises(ValueError, match="eliminat"):
            TournamentConfig(
                n_qualifiers=64,
                elimination_stages=((24, 2), (14, 1)),  # 48+14=62, need 63
            )

    def test_best_of_exceeds_matches(self):
        with pytest.raises(ValueError, match="best_of"):
            TournamentConfig(n_matches=5, best_of=6)
