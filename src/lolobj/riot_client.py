"""Thin Riot API client for Match-V5 ingestion.

Handles:
- regional vs platform routing
- development-key rate limits (20 req / 1s, 100 req / 120s) via a sliding-window limiter
- retries with backoff on 429 / 5xx, honoring Retry-After when present

Only the endpoints we actually need for objective-setup analytics are exposed.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from .config import RiotConfig, load_riot_config

logger = logging.getLogger(__name__)

# Maps platform routing value → regional routing value.
PLATFORM_TO_REGION: dict[str, str] = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe",  "eun1": "europe",  "tr1": "europe",  "ru": "europe",
    "kr":  "asia",     "jp1": "asia",
    "oc1": "sea",
}


class RateLimitError(Exception):
    """Raised when a 429 is received and ``raise_on_rate_limit`` is set."""
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"rate limited; retry after {retry_after:.1f}s")


@dataclass
class _RateLimit:
    """Sliding-window rate limit. Records request timestamps and blocks if full."""

    max_requests: int
    window_seconds: float
    _events: deque = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._events = deque()

    def wait(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if len(self._events) >= self.max_requests:
            sleep_for = self._events[0] + self.window_seconds - now
            if sleep_for > 0:
                time.sleep(sleep_for)

    def record(self) -> None:
        self._events.append(time.monotonic())


class RiotClient:
    """Minimal Match-V5 + Account-V1 client.

    Defaults are tuned for a Riot personal development key. If you upgrade to a
    production key, pass higher limits via ``rate_limits``.
    """

    DEFAULT_LIMITS = (
        (20, 1.0),     # 20 requests / second
        (100, 120.0),  # 100 requests / 2 minutes
    )

    def __init__(
        self,
        config: RiotConfig | None = None,
        rate_limits: Iterable[tuple[int, float]] | None = None,
        max_retries: int = 5,
        session: requests.Session | None = None,
        raise_on_rate_limit: bool = False,
    ) -> None:
        self.config = config or load_riot_config()
        self.session = session or requests.Session()
        self.session.headers.update({"X-Riot-Token": self.config.api_key})
        limits = tuple(rate_limits) if rate_limits is not None else self.DEFAULT_LIMITS
        self._limits = [_RateLimit(n, w) for n, w in limits]
        self.max_retries = max_retries
        self.raise_on_rate_limit = raise_on_rate_limit

    # ------------------------------------------------------------------ HTTP

    def _request(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self.max_retries):
            for limit in self._limits:
                limit.wait()
            for limit in self._limits:
                limit.record()
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "1"))
                if self.raise_on_rate_limit:
                    raise RateLimitError(retry_after)
                logger.warning("429 from Riot, sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            if 500 <= resp.status_code < 600:
                backoff = 2 ** attempt
                logger.warning("Riot %s, retry in %ds", resp.status_code, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Riot request failed after {self.max_retries} retries: {url}")

    # -------------------------------------------------------- Account / Match

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> dict[str, Any]:
        url = f"{self.config.region_host}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._request(url)

    def get_match_ids_by_puuid(
        self,
        puuid: str,
        *,
        start: int = 0,
        count: int = 20,
        queue: int | None = None,
        match_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        url = f"{self.config.region_host}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: dict[str, Any] = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        if match_type is not None:
            params["type"] = match_type
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._request(url, params=params)

    def get_match(self, match_id: str) -> dict[str, Any]:
        url = f"{self.config.region_host}/lol/match/v5/matches/{match_id}"
        return self._request(url)

    def get_match_timeline(self, match_id: str) -> dict[str, Any]:
        url = f"{self.config.region_host}/lol/match/v5/matches/{match_id}/timeline"
        return self._request(url)

    # ---------------------------------------------------------- Summoner / League

    def get_summoner_by_id(self, summoner_id: str) -> dict[str, Any]:
        """Summoner-V4: resolve summonerId → puuid (and other fields)."""
        url = f"{self.config.platform_host}/lol/summoner/v4/summoners/{summoner_id}"
        return self._request(url)

    def get_league_entries(
        self,
        tier: str,
        division: str,
        queue: str = "RANKED_SOLO_5x5",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """League-V4 entries for a tier/division page (Iron–Diamond only).

        ``tier`` is uppercase, e.g. ``"GOLD"``.
        ``division`` is ``"I"`` / ``"II"`` / ``"III"`` / ``"IV"``.
        Returns a list of league-entry dicts, each containing ``summonerId``.
        """
        url = f"{self.config.platform_host}/lol/league/v4/entries/{queue}/{tier}/{division}"
        return self._request(url, params={"page": page})

    def get_apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> dict[str, Any]:
        """League-V4 full league for Master / Grandmaster / Challenger.

        Returns a league object whose ``entries`` list contains ``summonerId``.
        """
        tier_lower = tier.lower()
        if tier_lower not in ("master", "grandmaster", "challenger"):
            raise ValueError(f"get_apex_league only supports master/grandmaster/challenger, got {tier!r}")
        url = f"{self.config.platform_host}/lol/league/v4/{tier_lower}leagues/by-queue/{queue}"
        return self._request(url)
