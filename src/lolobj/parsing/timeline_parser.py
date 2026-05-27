"""Parse Match-V5 timeline JSON into Python dataclasses.

Riot's MATCH-V5 timeline shape (relevant pieces):

    {
      "info": {
        "frameInterval": 60000,           # ms, usually 60_000
        "frames": [
          {
            "timestamp": <ms>,
            "participantFrames": {
              "1": { "participantId": 1, "level": 6, "totalGold": 4321,
                      "currentGold": 100, "xp": 5000, "minionsKilled": 50,
                      "jungleMinionsKilled": 10, "position": {"x": 1234, "y": 5678}, ... },
              ...
              "10": { ... }
            },
            "events": [
              {"type": "CHAMPION_KILL", "timestamp": ..., "killerId": 3, "victimId": 6, ...},
              {"type": "ELITE_MONSTER_KILL", "timestamp": ..., "killerId": 3,
                "killerTeamId": 100, "monsterType": "DRAGON", "monsterSubType": "WATER_DRAGON",
                "position": {...}, "assistingParticipantIds": [...]},
              {"type": "BUILDING_KILL", "timestamp": ..., "killerId": 3,
                "buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", ...},
              {"type": "WARD_PLACED" | "WARD_KILL", ...},
              {"type": "TURRET_PLATE_DESTROYED", ...},
              ...
            ]
          },
          ...
        ]
      }
    }

We keep raw event dicts (they vary by event type) and provide a few typed wrappers
for the frames and participant frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Position:
    x: int
    y: int

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Position | None":
        if not d:
            return None
        return cls(int(d.get("x", 0)), int(d.get("y", 0)))


@dataclass
class ParticipantFrame:
    participant_id: int
    timestamp_ms: int
    total_gold: int
    current_gold: int
    level: int
    xp: int
    minions_killed: int
    jungle_minions_killed: int
    position: Position | None

    @classmethod
    def from_dict(cls, pf: dict[str, Any], timestamp_ms: int) -> "ParticipantFrame":
        return cls(
            participant_id=int(pf.get("participantId", 0)),
            timestamp_ms=timestamp_ms,
            total_gold=int(pf.get("totalGold", 0)),
            current_gold=int(pf.get("currentGold", 0)),
            level=int(pf.get("level", 1)),
            xp=int(pf.get("xp", 0)),
            minions_killed=int(pf.get("minionsKilled", 0)),
            jungle_minions_killed=int(pf.get("jungleMinionsKilled", 0)),
            position=Position.from_dict(pf.get("position")),
        )


@dataclass
class Frame:
    timestamp_ms: int
    participant_frames: dict[int, ParticipantFrame]
    events: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Frame":
        ts = int(d.get("timestamp", 0))
        pfs_raw = d.get("participantFrames", {}) or {}
        pfs = {
            int(k): ParticipantFrame.from_dict(v, ts) for k, v in pfs_raw.items()
        }
        events = list(d.get("events", []) or [])
        return cls(timestamp_ms=ts, participant_frames=pfs, events=events)


@dataclass
class Timeline:
    match_id: str
    frame_interval_ms: int
    frames: list[Frame]
    # convenience: participantId -> teamId; derived from match info (passed separately)
    participant_team: dict[int, int] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return self.frames[-1].timestamp_ms if self.frames else 0

    def iter_events(self) -> Iterator[dict[str, Any]]:
        for f in self.frames:
            for e in f.events:
                yield e


def parse_timeline(timeline_payload: dict[str, Any], match_payload: dict[str, Any] | None = None) -> Timeline:
    """Parse a MATCH-V5 timeline payload into a :class:`Timeline`.

    If ``match_payload`` is provided, populates ``participant_team`` from it.
    """
    info = timeline_payload.get("info", timeline_payload)
    metadata = timeline_payload.get("metadata", {}) or {}
    match_id = metadata.get("matchId", "")
    frame_interval = int(info.get("frameInterval", 60_000))
    frames = [Frame.from_dict(f) for f in info.get("frames", []) or []]
    participant_team: dict[int, int] = {}
    if match_payload is not None:
        m_info = match_payload.get("info", {}) or {}
        for p in m_info.get("participants", []) or []:
            participant_team[int(p["participantId"])] = int(p["teamId"])
    return Timeline(
        match_id=match_id,
        frame_interval_ms=frame_interval,
        frames=frames,
        participant_team=participant_team,
    )
