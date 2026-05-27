"""Project configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class RiotConfig:
    api_key: str
    platform: str  # e.g. "na1" — used for SUMMONER-V4 etc.
    region: str    # e.g. "americas" — used for MATCH-V5, ACCOUNT-V1

    @property
    def platform_host(self) -> str:
        return f"https://{self.platform}.api.riotgames.com"

    @property
    def region_host(self) -> str:
        return f"https://{self.region}.api.riotgames.com"


def load_riot_config() -> RiotConfig:
    """Load Riot API credentials/routing from .env (if present) and environment."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("RIOT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "RIOT_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    platform = os.environ.get("RIOT_PLATFORM", "na1").strip().lower()
    region = os.environ.get("RIOT_REGION", "americas").strip().lower()
    return RiotConfig(api_key=api_key, platform=platform, region=region)


def ensure_data_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
