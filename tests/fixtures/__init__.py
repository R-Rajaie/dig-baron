"""Synthetic Riot API fixtures for unit tests."""

from __future__ import annotations

from typing import Any

# Dragon pit approximate position in game units
DRAGON_POS = {"x": 9866, "y": 4414}
# Within 2500 units of dragon pit (~424 units away)
NEARBY_POS = {"x": 9500, "y": 4200}
# Far from dragon pit (~7010 units away)
FAR_POS = {"x": 3000, "y": 3000}


def make_participant(
    pid: int,
    team_id: int,
    position: str,
    champion: str = "Ahri",
) -> dict[str, Any]:
    return {
        "participantId": pid,
        "teamId": team_id,
        "teamPosition": position,
        "individualPosition": position,
        "championName": champion,
    }


def make_match(match_id: str = "NA1_TEST001", patch: str = "14.1.1") -> dict[str, Any]:
    """Minimal match-V5 payload with 10 participants (5 per team, all roles filled)."""
    participants = [
        make_participant(1, 100, "TOP", "Jax"),
        make_participant(2, 100, "JUNGLE", "Vi"),
        make_participant(3, 100, "MIDDLE", "Ahri"),
        make_participant(4, 100, "BOTTOM", "Jinx"),
        make_participant(5, 100, "UTILITY", "Thresh"),
        make_participant(6, 200, "TOP", "Fiora"),
        make_participant(7, 200, "JUNGLE", "Graves"),
        make_participant(8, 200, "MIDDLE", "Zed"),
        make_participant(9, 200, "BOTTOM", "Caitlyn"),
        make_participant(10, 200, "UTILITY", "Lulu"),
    ]
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameVersion": patch,
            "participants": participants,
            "teams": [
                {"teamId": 100, "objectives": {}},
                {"teamId": 200, "objectives": {}},
            ],
        },
    }


def make_participant_frame(
    pid: int,
    total_gold: int = 3000,
    level: int = 7,
    position: dict[str, int] | None = None,
) -> dict[str, Any]:
    pos = position or {"x": 0, "y": 0}
    return {
        "participantId": pid,
        "totalGold": total_gold,
        "currentGold": 500,
        "level": level,
        "xp": 5000,
        "minionsKilled": 50,
        "jungleMinionsKilled": 20,
        "position": pos,
    }


def make_frame(
    timestamp_ms: int,
    participant_positions: dict[int, dict[str, int]] | None = None,
    events: list[dict[str, Any]] | None = None,
    gold_by_pid: dict[int, int] | None = None,
    level_by_pid: dict[int, int] | None = None,
) -> dict[str, Any]:
    positions = participant_positions or {}
    golds = gold_by_pid or {}
    levels = level_by_pid or {}
    pfs: dict[str, Any] = {}
    for pid in range(1, 11):
        pfs[str(pid)] = make_participant_frame(
            pid=pid,
            total_gold=golds.get(pid, 3000),
            level=levels.get(pid, 7),
            position=positions.get(pid),
        )
    return {
        "timestamp": timestamp_ms,
        "participantFrames": pfs,
        "events": events or [],
    }


def dragon_kill_event(
    timestamp_ms: int,
    killer_team_id: int,
    killer_id: int = 2,
    assisting: list[int] | None = None,
    subtype: str = "WATER_DRAGON",
) -> dict[str, Any]:
    return {
        "type": "ELITE_MONSTER_KILL",
        "timestamp": timestamp_ms,
        "killerId": killer_id,
        "killerTeamId": killer_team_id,
        "monsterType": "DRAGON",
        "monsterSubType": subtype,
        "position": DRAGON_POS,
        "assistingParticipantIds": assisting or [],
    }


def champion_kill_event(
    timestamp_ms: int,
    killer_id: int,
    victim_id: int,
    assisting: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "type": "CHAMPION_KILL",
        "timestamp": timestamp_ms,
        "killerId": killer_id,
        "victimId": victim_id,
        "assistingParticipantIds": assisting or [],
        "position": NEARBY_POS,
    }


def make_minimal_timeline(
    match_id: str = "NA1_TEST001",
    frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "frameInterval": 60000,
            "frames": frames or [],
        },
    }
