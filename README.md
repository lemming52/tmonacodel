# tmonacodel

Monte Carlo simulator for Trackmania tournament formats. Runs thousands of simulated tournaments to estimate each player's probability of qualification, expected points, and finish distribution.

I wanted to build this (read; tell claude to build this) because:
1. The extremely top heavy point distribution of the points per race seemed to suggest that a couple of good results would almost certainly be enough to qualify.
2. The format is easy to understand and break down into stages. Firstly we can simulate a round overall, then we can simulate each round of that round, and possibly even expand into modelling player ability with ELO like systems, reading live data based on trackmania APIs, sophisticated round simulation using actual times, player stats and all sorts.
3. The progression of the model sophistication could be done incrementally, a fully random monte-carlo is (although only slightly representative) fairly useful as it suggests a lower bound, but as we improve the sophistication and respond to the actual results it can be adjusted to be more representative.
4. Possibly as a result of this i might be able to find an interesting if not necessairly useful way of modelling each player.

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.13+.

## Usage

```python
from tmonacodel import run_monte_carlo

results = run_monte_carlo()
print(results.summary_dataframe())
```

### Custom config

```python
from tmonacodel import run_monte_carlo
from tmonacodel.config import TournamentConfig

cfg = TournamentConfig(
    n_players=128,
    n_qualifiers=64,
    n_matches=10,
    best_of=5,
    n_simulations=50_000,
    random_seed=None,      # None = non-deterministic
    top_n_cutoff=16,       # threshold for "qualified for next stage"
)
results = run_monte_carlo(config=cfg)
```

### Results API

```python
df = results.summary_dataframe()   # per-player stats (median points, p90, prob_top_N, …)
results.plot_points_distribution() # histogram per player
```

## Running tests

```bash
pytest
```

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | `TournamentConfig` frozen dataclass — all knobs in one place |
| `racer.py` | `Racer` dataclass + `make_racer_pool` |
| `scoring.py` | Points table, built once and reused |
| `elimination.py` | Hot path: one `rng.permutation` + fancy-index scatter per match |
| `match.py` | `qualify_players` + `simulate_match` |
| `season.py` | Best-N-of-M rule via numpy sort |
| `simulation.py` | Monte Carlo driver with a single seeded RNG |
| `aggregation.py` | `SimulationResults` — public results API |

## Tournament format

The default config models a 120-player field:

- **Qualification**: 64 players advance from 128 via head-to-head matches
- **Match format**: best 5 of 10 per player
- **Elimination stages**: 24 rounds eliminating 2 per round, then 15 rounds eliminating 1
- **Simulations**: 10,000 runs per call (adjust via `n_simulations`)

## Name

> `tmonacodel` is a terrible portmanteau of TrackMania (TM), Monaco (Monte Carlo GP) and Model