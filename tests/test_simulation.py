import numpy as np
import pytest

from tmonacodel import TournamentConfig, run_monte_carlo, REAL_PLAYERS, RealCupResults


class TestRunMonteCarlo:
    def test_returns_results(self, default_config):
        results = run_monte_carlo(TournamentConfig(n_simulations=50, random_seed=0))
        assert results is not None

    def test_all_tournament_points_shape(self, default_config):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_tournament_points.shape == (50, cfg.n_players)

    def test_all_tournament_ranks_shape(self, default_config):
        cfg = TournamentConfig(n_simulations=50, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_tournament_ranks.shape == (50, cfg.n_players)

    def test_reproducible(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=42)
        r1 = run_monte_carlo(cfg)
        r2 = run_monte_carlo(cfg)
        np.testing.assert_array_equal(r1.all_tournament_points, r2.all_tournament_points)

    def test_different_seeds_differ(self):
        r1 = run_monte_carlo(TournamentConfig(n_simulations=20, random_seed=1))
        r2 = run_monte_carlo(TournamentConfig(n_simulations=20, random_seed=2))
        assert not np.array_equal(r1.all_tournament_points, r2.all_tournament_points)

    def test_ranks_range(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        assert results.all_tournament_ranks.min() >= 1
        assert results.all_tournament_ranks.max() <= cfg.n_players

    def test_summary_dataframe_rows(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        df = results.summary_dataframe()
        assert len(df) == cfg.n_players

    def test_summary_dataframe_columns(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=0)
        results = run_monte_carlo(cfg)
        df = results.summary_dataframe()
        for col in ["player_id", "name", "mean_rank", "median_rank", "prob_top_N"]:
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
        assert results.all_tournament_points.shape[0] == 10

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

    def test_real_results_accepted(self):
        cfg = TournamentConfig(n_simulations=10, random_seed=0)
        real: RealCupResults = [{"Mudda": 1000, "Binkss": 700}, None, None]
        results = run_monte_carlo(cfg, player_data=REAL_PLAYERS, real_cup_results=real)
        assert results.all_tournament_points.shape == (10, cfg.n_players)
        assert results.all_tournament_ranks.shape == (10, cfg.n_players)

    def test_real_results_reproducible(self):
        cfg = TournamentConfig(n_simulations=20, random_seed=42)
        real: RealCupResults = [{"Mudda": 1000}, None]
        r1 = run_monte_carlo(cfg, player_data=REAL_PLAYERS, real_cup_results=real)
        r2 = run_monte_carlo(cfg, player_data=REAL_PLAYERS, real_cup_results=real)
        np.testing.assert_array_equal(r1.all_tournament_points, r2.all_tournament_points)

    def test_real_results_all_fixed_ignores_rng(self):
        """When every cup is fixed, two different seeds produce identical output."""
        cfg = TournamentConfig(n_simulations=10, random_seed=1)
        real: RealCupResults = [{"Mudda": 500}] * cfg.n_cups
        r1 = run_monte_carlo(cfg, player_data=REAL_PLAYERS, real_cup_results=real)

        cfg2 = TournamentConfig(n_simulations=10, random_seed=999)
        r2 = run_monte_carlo(cfg2, player_data=REAL_PLAYERS, real_cup_results=real)
        np.testing.assert_array_equal(r1.all_tournament_points, r2.all_tournament_points)

    def test_real_results_shorter_than_n_cups(self):
        """Trailing cups are simulated, so output should vary across seeds."""
        real: RealCupResults = [{"Mudda": 1000}]  # only 1 of n_cups fixed
        r1 = run_monte_carlo(
            TournamentConfig(n_simulations=20, random_seed=1),
            player_data=REAL_PLAYERS, real_cup_results=real,
        )
        r2 = run_monte_carlo(
            TournamentConfig(n_simulations=20, random_seed=2),
            player_data=REAL_PLAYERS, real_cup_results=real,
        )
        assert not np.array_equal(r1.all_tournament_points, r2.all_tournament_points)

    def test_unknown_names_ignored(self):
        """Unknown player names in real_cup_results raise no error."""
        cfg = TournamentConfig(n_simulations=5, random_seed=0)
        real: RealCupResults = [{"DoesNotExist": 9999, "AlsoFake": 1234}]
        results = run_monte_carlo(cfg, player_data=REAL_PLAYERS, real_cup_results=real)
        assert results is not None

    def test_real_results_none_same_as_omitted(self):
        """Passing real_cup_results=None is identical to not passing it."""
        cfg = TournamentConfig(n_simulations=20, random_seed=7)
        r1 = run_monte_carlo(cfg)
        r2 = run_monte_carlo(cfg, real_cup_results=None)
        np.testing.assert_array_equal(r1.all_tournament_points, r2.all_tournament_points)
