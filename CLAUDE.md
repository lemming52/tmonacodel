# tmonacodel — Trackmania Monte Carlo Simulator

## Setup

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Quick smoke test

```bash
python -c "from tmonacodel import run_monte_carlo; r = run_monte_carlo(); print(r.summary_dataframe().head())"
```

## Architecture

- `config.py` — all knobs in `TournamentConfig` (frozen dataclass)
- `player.py` — `Player` dataclass + `make_player_pool`; extension point for skill stats
- `scoring.py` — points table built once, reused across all sims
- `race.py` — hot path: one `rng.permutation` + fancy-index scatter (39 elimination rounds)
- `cup.py` — `qualify_players` + `simulate_cup` (qualification + race for one COTW)
- `tournament.py` — best-N-of-M rule via numpy sort (one Elite Cup)
- `simulation.py` — Monte Carlo driver, single seeded RNG
- `aggregation.py` — `SimulationResults` public API

## Terminology

| Official name | Code term |
|---|---|
| Cup of the Week (COTW) | `race` |
| Elite Cup | `tournament` |
| Round within a COTW | `round` |
| Individual competitor | `player` |

## Extension points

| Feature | Files to change |
|---|---|
| Per-player skill affects elimination | `player.py`, `race.py`, `simulation.py` |
| Skill affects qualification | `cup.py` |
| Player evolution across cups | `tournament.py`, `player.py` |
| Load players from external dataset | `player.py` / new `data_loader.py` |
