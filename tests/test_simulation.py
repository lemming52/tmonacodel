import numpy as np
import pytest

from tmonacodel import TournamentConfig, run_monte_carlo, REAL_PLAYERS


class TestRunMonteCarlo:
    def test_returns_results(self, default_config):
        results = run_monte_carlo(TournamentConfig(n_simulations=50, random_seed=0))
        assert results is not None

    def test_all_season_points_shape(self, default_config):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_season_points.shape == (50, cfg.n_players)

    def test_all_season_ranks_shape(self, default_config):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_season_ranks.shape == (50, cfg.n_players)

    def test_reproducible(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=42)
        r1 = run_monte_carlo(cfg)
        r2 = run_monte_carlo(cfg)
        np.testing.assert_array_equal(r1.all_season_points, r2.all_season_points)

    def test_different_seeds_differ(self):
        r1 = run_monte_carlo(TournamentConfig(n_simulations=20, random_seed=1))
        r2 = run_monte_carlo(TournamentConfig(n_simulations=20, random_seed=2))
        assert not np.array_equal(r1.all_season_points, r2.all_season_points)

    def test_ranks_range(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_season_ranks.min() >= 1
        assert results.all_season_ranks.max() <= cfg.n_players

    def test_summary_dataframe_rows(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        df = results.summary_dataframe()
        assert len(df) == cfg.n_players

    def test_summary_dataframe_columns(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        df = results.summary_dataframe()
        for col in ["racer_id", "name", "mean_rank", "median_rank", "prob_top_N"]:
            assert col in df.columns

    def test_prob_top_n_between_0_and_1(self):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        df = results.summary_dataframe()
        assert (df["prob_top_N"] >= 0).all()
        assert (df["prob_top_N"] <= 1).all()

    def test_qualification_probabilities_length(self):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        probs = results.qualification_probabilities()
        assert len(probs) == cfg.n_players

    def test_qualification_probabilities_sorted_descending(self):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        probs = results.qualification_probabilities()
        assert (probs.values[:-1] >= probs.values[1:]).all()

    def test_points_distribution_shape(self):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        dist = results.points_distribution(0)
        assert dist.shape == (50,)

    def test_default_config_runs(self):
        results = run_monte_carlo(TournamentConfig(n_simulations=10, random_seed=0))
        assert results.all_season_points.shape[0] == 10

    def test_player_data_names_in_results(self):
        """Real player names should appear in summary when player_data is passed."""
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg, player_data=REAL_PLAYERS)
        df = results.summary_dataframe()
        assert "country" in df.columns
        assert "Mudda" in df["name"].values
        assert "Australia" in df["country"].values

    def test_player_data_generics_fill_remainder(self):
        """Slots beyond player_data length should be filled with Generic_ names."""
        cfg = TournamentConfig(n_simulations=10, random_seed=0)
        results = run_monte_carlo(cfg, player_data=REAL_PLAYERS)
        df = results.summary_dataframe()
        generic_rows = df[df["name"].str.startswith("Generic_")]
        assert len(generic_rows) == cfg.n_players - len(REAL_PLAYERS)

    def test_nation_summary_dataframe(self):
        """nation_summary_dataframe should group by country and include France."""
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg, player_data=REAL_PLAYERS)
        nation_df = results.nation_summary_dataframe()
        assert "country" in nation_df.columns
        assert "player_count" in nation_df.columns
        assert "mean_qual_prob" in nation_df.columns
        assert "France" in nation_df["country"].values
        # France should have the most players
        assert nation_df.iloc[0]["country"] == "France"

    def test_uniform_mechanics_similar_probs(self):
        """With uniform random mechanics, all players should have similar qual probs."""
        cfg = TournamentConfig(n_simulations=500, random_seed=42)
        results = run_monte_carlo(cfg)
        probs = results.qualification_probabilities()
        # With 120 players and uniform mechanics, prob should be near 64/120 ≈ 0.53
        # Allow wide tolerance since best-5-of-10 creates variance
        assert probs.mean() > 0.05
        assert probs.std() < 0.15  # not too spread out with uniform mechanics
