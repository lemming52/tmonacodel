from __future__ import annotations
import numpy as np

from .config import TournamentConfig


def build_points_table(n_qualifiers: int) -> np.ndarray:
    """
    Returns array of shape (n_qualifiers + 1,) where:
      index 0 = DNQ = 0 points
      index k = points for finishing k-th in the match (1-based)

    Points spec (for n_qualifiers=64):
      1st=1000, 2nd=700, 3rd=600, 4th=500, 5th=400, 6th=300, 7th=200, 8th=100
      9th=96, 10th=92, 11th=88, 12th=84, 13th=80, 14th=76, 15th=72, 16th=68
      Grouped pairs from 17-18 down to 63-64 (step -4 per pair)
    """
    pts = np.zeros(n_qualifiers + 1, dtype=np.int64)

    # Top 8 (indices 1-8)
    top8 = [1000, 700, 600, 500, 400, 300, 200, 100]
    for i, p in enumerate(top8, start=1):
        if i <= n_qualifiers:
            pts[i] = p

    # 9th through 16th: 96 down to 68 by -4 (indices 9-16)
    val = 96
    for i in range(9, 17):
        if i <= n_qualifiers:
            pts[i] = val
            val -= 4

    # Grouped pairs from position 17 onward.
    # Spec: 17-18=64 ... 33-34=32 (step -4), then 35-36=30 ... 63-64=2 (step -2)
    pair_values = list(range(64, 31, -4)) + list(range(30, 0, -2))
    pos = 17
    for pair_val in pair_values:
        if pos > n_qualifiers:
            break
        pts[pos] = pair_val
        if pos + 1 <= n_qualifiers:
            pts[pos + 1] = pair_val
        pos += 2

    return pts


def build_finish_position_lookup(config: TournamentConfig) -> np.ndarray:
    """
    Returns array of shape (n_qualifiers,) mapping permutation index → finish position.

    The permutation index represents the order in which players are eliminated:
      - index 0 is eliminated first (worst finish)
      - index n_qualifiers-1 is the last standing (1st place)

    Players eliminated in the same round share the same finish position.
    """
    n = config.n_qualifiers
    finish_positions = np.empty(n, dtype=np.int64)

    # Build from the end: last standing = position 1
    # Work backwards through elimination stages
    remaining = n
    pos = 1  # current finish position being assigned (from best to worst)

    # We need to assign positions from 1st (last eliminated) to nth (first eliminated).
    # Collect all rounds as (finish_position, count) from best to worst.
    rounds: list[tuple[int, int]] = []

    # Process stages in order: each stage eliminates some players
    # Stage 1: 24 rounds × 2 eliminated = 48 players (64→16)
    # Stage 2: 15 rounds × 1 eliminated = 15 players (16→1)
    # The 1 survivor gets 1st place.

    # Start with the survivor: finish position 1
    rounds.append((1, 1))  # 1 player finishes 1st
    pos = 2
    remaining = 1

    # Process stages in reverse (last stage first = best finishes)
    for n_rounds, n_elim in reversed(config.elimination_stages):
        for _ in range(n_rounds):
            rounds.append((pos, n_elim))
            pos += n_elim
            remaining += n_elim

    # rounds is now from best finish (1st) to worst finish
    # The permutation index is elimination order: 0 = first eliminated (worst), n-1 = survivor (best)
    # Map permutation index → finish position
    # perm_idx 0..(n_elim-1) in last round → worst finish
    # perm_idx n-1 → 1st place

    # Build lookup: assign finish positions in order from survivor outward
    # rounds[0] = (1, 1): the survivor at perm_idx = n-1
    # rounds[1] = last round of last stage, at perm_idx n-2 down to n-2-(n_elim-1)
    # etc.
    lookup = np.empty(n, dtype=np.int64)
    perm_idx = n - 1  # start from survivor
    for finish_pos, count in rounds:
        for j in range(count):
            lookup[perm_idx] = finish_pos + j if count == 1 else finish_pos
            # For ties (count > 1), all share finish_pos
            perm_idx -= 1
        # Adjust for paired positions
        # Actually all n_elim players eliminated in same round share same finish_pos

    # Reassign correctly: ties share finish_pos (not finish_pos + j)
    lookup2 = np.empty(n, dtype=np.int64)
    perm_idx = n - 1
    for finish_pos, count in rounds:
        for _ in range(count):
            lookup2[perm_idx] = finish_pos
            perm_idx -= 1

    return lookup2
