from __future__ import annotations
import numpy as np
import pytest

from tmonacodel.player import derive_skill, Player


class TestDeriveSkill:
    def test_higher_points_higher_skill(self):
        results = [
            {"Alice": 1000, "Bob": 100},
            {"Alice": 900, "Bob": 200},
        ]
        skill_map, _ = derive_skill(results, [])
        assert skill_map["Alice"] > skill_map["Bob"]

    def test_unknown_names_not_in_map(self):
        results = [{"Alice": 800}]
        skill_map, _ = derive_skill(results, [])
        assert "Bob" not in skill_map
        assert "Alice" in skill_map

    def test_generic_skill_at_25th_percentile(self):
        results = [{"A": 1000, "B": 700, "C": 400, "D": 100}]
        skill_map, generic_skill = derive_skill(results, [])
        skill_values = list(skill_map.values())
        expected_p25 = float(np.percentile(skill_values, 25))
        assert abs(generic_skill - expected_p25) < 1e-9

    def test_no_zero_weights(self):
        results = [{"A": 0, "B": 500, "C": 1000}]
        skill_map, _ = derive_skill(results, [])
        assert all(v >= 0.01 for v in skill_map.values())

    def test_empty_results_returns_empty_map(self):
        skill_map, generic_skill = derive_skill([], [])
        assert skill_map == {}
        assert generic_skill == 1.0

    def test_skill_clipped_at_1(self):
        results = [{"Elite": 1000}]
        skill_map, _ = derive_skill(results, [])
        assert skill_map["Elite"] <= 1.0

    def test_multiple_cups_averaged(self):
        # Alice appears in 2 cups, mean should be (1000 + 0) / 2 = 500 → skill 0.5
        # But derive_skill only counts cups where player appears, so Alice: mean(1000) = 1000
        # Bob: appears in cup1 with 400
        results = [{"Alice": 1000}, {"Bob": 400}]
        skill_map, _ = derive_skill(results, [])
        assert abs(skill_map["Alice"] - 1.0) < 1e-9
        assert abs(skill_map["Bob"] - 0.4) < 1e-9
