"""Detect objective takes (Dragon, Herald, Baron, Voidgrubs) from a parsed timeline.

Match-V5 emits ``ELITE_MONSTER_KILL`` events for these. Schema:

    {
      "type": "ELITE_MONSTER_KILL",
      "timestamp": <ms>,
      "killerId": <participantId, 0 if executed by minion/turret>,
      "killerTeamId": 100 | 200,
      "monsterType": "DRAGON" | "RIFTHERALD" | "BARON_NASHOR" | "HORDE",
      "monsterSubType": "FIRE_DRAGON" | "AIR_DRAGON" | "EARTH_DRAGON" |
                         "WATER_DRAGON" | "HEXTECH_DRAGON" | "CHEMTECH_DRAGON" |
                         "ELDER_DRAGON" | (absent for non-dragons),
      "position": {"x": ..., "y": ...},
      "assistingParticipantIds": [...]
    }

For dragons we attach an ordinal: which Nth dragon kill in the match this is (global,
not per-team — dragons are unique on the map). For voidgrubs (HORDE) each kill is its
own event; ordinal is the count from match start. Baron and Herald similarly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .timeline_parser import Timeline

DRAGON = "DRAGON"
ELDER_DRAGON_SUBTYPE = "ELDER_DRAGON"
RIFTHERALD = "RIFTHERALD"
BARON = "BARON_NASHOR"
HORDE = "HORDE"  # voidgrubs

OBJECTIVE_MONSTERS = {DRAGON, RIFTHERALD, BARON, HORDE}


@dataclass
class ObjectiveEvent:
    """A single objective-take event with derived ordinal."""

    match_id: str
    timestamp_ms: int
    monster_type: str          # DRAGON, RIFTHERALD, BARON_NASHOR, HORDE
    monster_subtype: str | None  # dragon element, or None
    killer_team_id: int        # 100 (blue) or 200 (red)
    killer_id: int             # participantId, 0 if structure executed it
    ordinal: int               # 1-indexed match-global ordinal for this objective_type
    position: tuple[int, int] | None
    assisting_participant_ids: list[int]
    # For dragons, "is_soul" = ordinal >= 4 (the 4th dragon awards soul). "is_elder" = subtype is ELDER.
    is_elder: bool
    raw: dict[str, Any]

    @property
    def objective_instance(self) -> str:
        """Stable string key like 'dragon_1', 'baron_1', 'herald_2', 'voidgrub_1'."""
        prefix = {
            DRAGON: "dragon",
            RIFTHERALD: "herald",
            BARON: "baron",
            HORDE: "voidgrub",
        }[self.monster_type]
        if self.is_elder:
            prefix = "elder"
        return f"{prefix}_{self.ordinal}"


def extract_objective_events(timeline: Timeline) -> list[ObjectiveEvent]:
    """Walk a parsed Timeline and produce ObjectiveEvent rows in chronological order.

    Ordinals are 1-indexed per monster_type (with Elder Dragon counted separately).
    """
    counters: dict[str, int] = {}
    out: list[ObjectiveEvent] = []
    for ev in timeline.iter_events():
        if ev.get("type") != "ELITE_MONSTER_KILL":
            continue
        mtype = ev.get("monsterType")
        if mtype not in OBJECTIVE_MONSTERS:
            continue
        subtype = ev.get("monsterSubType")
        is_elder = mtype == DRAGON and subtype == ELDER_DRAGON_SUBTYPE
        bucket = "ELDER" if is_elder else mtype
        counters[bucket] = counters.get(bucket, 0) + 1
        pos = ev.get("position") or {}
        out.append(
            ObjectiveEvent(
                match_id=timeline.match_id,
                timestamp_ms=int(ev.get("timestamp", 0)),
                monster_type=mtype,
                monster_subtype=subtype,
                killer_team_id=int(ev.get("killerTeamId", 0)),
                killer_id=int(ev.get("killerId", 0)),
                ordinal=counters[bucket],
                position=(int(pos.get("x", 0)), int(pos.get("y", 0))) if pos else None,
                assisting_participant_ids=list(ev.get("assistingParticipantIds", []) or []),
                is_elder=is_elder,
                raw=ev,
            )
        )
    return out


def filter_dragons(events: Iterable[ObjectiveEvent], ordinals: Iterable[int] = (1, 2)) -> list[ObjectiveEvent]:
    """Convenience: keep Dragon ordinals in ``ordinals`` (default the first two)."""
    keep = set(ordinals)
    return [e for e in events if e.monster_type == DRAGON and not e.is_elder and e.ordinal in keep]
