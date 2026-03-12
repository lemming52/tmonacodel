from __future__ import annotations

# player_name → points earned in one cup
CupResultByName = dict[str, int]

# One entry per cup slot; None means "simulate this cup"
RealCupResults = list[CupResultByName | None]
