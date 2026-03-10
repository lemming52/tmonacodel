from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TournamentConfig
from .player import Player


@dataclass
class SimulationResults:
    config: TournamentConfig
    players: list[Player]
    all_tournament_points: np.ndarray  # shape (n_sims, n_players)
    all_tournament_ranks: np.ndarray   # shape (n_sims, n_players)

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Returns DataFrame with one row per player:
          player_id, name, mean_rank, median_rank,
          p10_points, p50_points, p90_points, prob_top_N
        """
        n = self.config.n_players
        rows = []
        for i in range(n):
            ranks = self.all_tournament_ranks[:, i]
            points = self.all_tournament_points[:, i]
            rows.append(
                {
                    "player_id": self.players[i].player_id,
                    "name": self.players[i].name,
                    "country": self.players[i].country,
                    "mean_rank": float(np.mean(ranks)),
                    "median_rank": float(np.median(ranks)),
                    "p10_points": int(np.percentile(points, 10)),
                    "p50_points": int(np.percentile(points, 50)),
                    "p90_points": int(np.percentile(points, 90)),
                    "prob_top_N": float(np.mean(ranks <= self.config.top_n_cutoff)),
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("mean_rank").reset_index(drop=True)
        return df

    def nation_summary_dataframe(self) -> pd.DataFrame:
        """
        Returns DataFrame grouped by country with player count and mean qual prob.
        Sorted descending by player count, then mean qual prob.
        """
        df = self.summary_dataframe()
        grouped = (
            df.groupby("country")
            .agg(
                player_count=("name", "count"),
                mean_qual_prob=("prob_top_N", "mean"),
                mean_rank=("mean_rank", "mean"),
            )
            .reset_index()
            .sort_values(["player_count", "mean_qual_prob"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return grouped

    def ewc_qualifying_score(self, top_n: int = 8) -> int:
        """
        Return PQ: the minimum score that would have placed in the top `top_n`
        in every simulation (guaranteed EWC qualifying threshold).
        """
        cutoffs = np.sort(self.all_tournament_points, axis=1)[:, -top_n]
        return int(np.max(cutoffs))

    def ewc_qualification_curve(self, top_n: int = 8, step: int = 50) -> pd.DataFrame:
        """
        For scores from PQ down to 0 (stepping by `step`), return the fraction
        of simulations in which that score would have placed top `top_n`.
        Stops early once prob_qualify reaches 0.
        """
        pq = self.ewc_qualifying_score(top_n)
        cutoffs = np.sort(self.all_tournament_points, axis=1)[:, -top_n]
        rows = []
        score = pq
        while score >= 0:
            prob = float(np.mean(cutoffs <= score))
            rows.append({"score": score, "pq_offset": pq - score, "prob_qualify": prob})
            if prob == 0.0:
                break
            score -= step
        return pd.DataFrame(rows)

    def nation_topping_scores(self) -> pd.DataFrame:
        """
        For each nation with ≥ 2 players, return the P90 points score
        needed to top the national leaderboard (i.e. the 90th percentile of
        the per-simulation national maximum, avoiding outlier distortion).
        """
        from collections import defaultdict
        nation_indices: dict[str, list[int]] = defaultdict(list)
        for p in self.players:
            if p.country:
                nation_indices[p.country].append(p.player_id)
        rows = []
        for country, indices in sorted(nation_indices.items()):
            if len(indices) < 2:
                continue
            nation_pts = self.all_tournament_points[:, indices]
            sim_maxes = nation_pts.max(axis=1)  # best score per simulation
            p90_to_top = int(np.percentile(sim_maxes, 90))
            rows.append({
                "country": country,
                "player_count": len(indices),
                "p90_score_to_top_nation": p90_to_top,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("p90_score_to_top_nation", ascending=False)
            .reset_index(drop=True)
        )

    def rank_points_profile(self) -> pd.DataFrame:
        """
        Returns DataFrame with one row per finishing rank (1 = best):
          rank, mean_points, p10_points, p90_points
        """
        n_sims = self.config.n_simulations
        order = np.argsort(self.all_tournament_ranks, axis=1)
        sorted_pts = self.all_tournament_points[np.arange(n_sims)[:, None], order]
        return pd.DataFrame({
            "rank": np.arange(1, self.config.n_players + 1),
            "mean_points": np.round(sorted_pts.mean(axis=0), 1),
            "p10_points": np.percentile(sorted_pts, 10, axis=0).astype(int),
            "p90_points": np.percentile(sorted_pts, 90, axis=0).astype(int),
        })

    def qualification_probabilities(self) -> pd.Series:
        """
        Returns Series indexed by player name, sorted descending by P(rank ≤ top_n_cutoff).
        """
        probs = {}
        for i in range(self.config.n_players):
            ranks = self.all_tournament_ranks[:, i]
            probs[self.players[i].name] = float(np.mean(ranks <= self.config.top_n_cutoff))
        series = pd.Series(probs, name="prob_qualify")
        return series.sort_values(ascending=False)

    def points_distribution(self, player_id: int) -> np.ndarray:
        """
        Returns array of shape (n_sims,) with tournament point totals for the given player_id.
        """
        return self.all_tournament_points[:, player_id].copy()
