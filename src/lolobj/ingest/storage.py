"""On-disk cache for raw Match-V5 match + timeline JSON.

Layout:
    data/raw/matches/<match_id>.json
    data/raw/timelines/<match_id>.json

Idempotent: ``has_match`` / ``has_timeline`` check existence before re-fetching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..config import RAW_DIR, ensure_data_dirs

MATCHES_SUBDIR = "matches"
TIMELINES_SUBDIR = "timelines"
RANK_META_SUBDIR = "rank_meta"


def _match_path(match_id: str, root: Path | None = None) -> Path:
    root = root or RAW_DIR
    return root / MATCHES_SUBDIR / f"{match_id}.json"


def _timeline_path(match_id: str, root: Path | None = None) -> Path:
    root = root or RAW_DIR
    return root / TIMELINES_SUBDIR / f"{match_id}.json"


def has_match(match_id: str, root: Path | None = None) -> bool:
    return _match_path(match_id, root).exists()


def has_timeline(match_id: str, root: Path | None = None) -> bool:
    return _timeline_path(match_id, root).exists()


def save_match(match_id: str, payload: dict[str, Any], root: Path | None = None) -> Path:
    ensure_data_dirs()
    path = _match_path(match_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def save_timeline(match_id: str, payload: dict[str, Any], root: Path | None = None) -> Path:
    ensure_data_dirs()
    path = _timeline_path(match_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_match(match_id: str, root: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads(_match_path(match_id, root).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt match JSON for {match_id}") from exc


def load_timeline(match_id: str, root: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads(_timeline_path(match_id, root).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt timeline JSON for {match_id}") from exc


_PUUID_BUCKETS_FILE = "puuid_buckets.json"


def _puuid_buckets_path(root: Path | None = None) -> Path:
    root = root or RAW_DIR
    return root / RANK_META_SUBDIR / _PUUID_BUCKETS_FILE


def save_puuid_buckets(new_entries: dict[str, str], root: Path | None = None) -> None:
    """Merge ``new_entries`` (puuid -> rank_bucket) into the persistent mapping."""
    path = _puuid_buckets_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(new_entries)
    path.write_text(json.dumps(existing), encoding="utf-8")


def load_puuid_buckets(root: Path | None = None) -> dict[str, str]:
    """Return the persistent puuid -> rank_bucket mapping (empty dict if none saved)."""
    path = _puuid_buckets_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_cached_match_ids(root: Path | None = None) -> Iterator[str]:
    """Yield match_ids that have BOTH match and timeline JSON cached."""
    root = root or RAW_DIR
    match_dir = root / MATCHES_SUBDIR
    timeline_dir = root / TIMELINES_SUBDIR
    if not match_dir.exists() or not timeline_dir.exists():
        return
    timelines = {p.stem for p in timeline_dir.glob("*.json")}
    for p in sorted(match_dir.glob("*.json")):
        if p.stem in timelines:
            yield p.stem
