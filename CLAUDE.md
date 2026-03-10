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
- `racer.py` — `Racer` dataclass + `make_racer_pool`; extension point for skill stats
- `scoring.py` — points table built once, reused across all sims
- `elimination.py` — hot path: one `rng.permutation` + fancy-index scatter per match
- `match.py` — `qualify_players` + `simulate_match`
- `season.py` — best-N-of-M rule via numpy sort
- `simulation.py` — Monte Carlo driver, single seeded RNG
- `aggregation.py` — `SimulationResults` public API

## Extension points

| Feature | Files to change |
|---|---|
| Per-racer skill affects elimination | `racer.py`, `elimination.py`, `simulation.py` |
| Skill affects qualification | `match.py` |
| Racer evolution across matches | `season.py`, `racer.py` |
| Load racers from external dataset | `racer.py` / new `data_loader.py` |
