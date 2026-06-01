"""Bulk-download matches sampled from multiple rank tiers and regions.

Strategy:
  1. For each requested region (na1, euw1, kr, …), pull player lists from
     League-V4 and resolve summonerId to PUUID via Summoner-V4.
  2. Build per-region download queues; each PUUID stays tied to its home region.
  3. Process queues in round-robin order.  When one region hits its 100 req/120s
     cap and raises RateLimitError, switch to the next region's queue instead of
     waiting.  Return to the cooling region once its window expires.

Using three regions (na1 + euw1 + kr) effectively triples throughput because
each region has an independent rate-limit counter for the same API key.

Usage examples:

    # Default: all four rank buckets, na1 only (single-region)
    python -m lolobj.ingest.seed_matches \\
        --tiers low mid high elite --players-per-tier 30 --matches-per-player 20

    # Three regions, higher throughput
    python -m lolobj.ingest.seed_matches \\
        --regions na1 euw1 kr \\
        --tiers low mid high elite --players-per-tier 30 --matches-per-player 20

    # Quick test: 5 players, 5 matches, Diamond only
    python -m lolobj.ingest.seed_matches \\
        --regions na1 --tiers high \\
        --players-per-tier 5 --matches-per-player 5 -v

Rank-bucket -> full division range (per CLAUDE.md):
    low   -> Iron IV through Silver I   (12 divisions)
    mid   -> Gold IV through Platinum I (8 divisions)
    high  -> Emerald IV through Diamond I (8 divisions)
    elite -> Master / Grandmaster / Challenger (apex leagues, no divisions)

Players are sampled uniformly across all divisions in the bucket: the division
list is shuffled on each run, one page per division is fetched, all candidates
are pooled and shuffled, then N are selected.  To add variety within a single
division a random page offset is also chosen (page 1 or 2), so you won't
always see the same players at the top of each leaderboard page.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import load_riot_config
from ..riot_client import PLATFORM_TO_REGION, RateLimitError, RiotClient, RiotConfig
from . import storage
from .download_matches import download_for_puuid

logger = logging.getLogger(__name__)

_DIVS = ("IV", "III", "II", "I")

# All tier+division combinations per bucket, ordered low to high within each bucket.
# Shuffled at runtime so sampling is spread uniformly across the full range.
BUCKET_ALL_DIVISIONS: dict[str, list[tuple[str, str | None]]] = {
    "low":   [(t, d) for t in ("IRON", "BRONZE", "SILVER") for d in _DIVS],
    "mid":   [(t, d) for t in ("GOLD", "PLATINUM") for d in _DIVS],
    "high":  [(t, d) for t in ("EMERALD", "DIAMOND") for d in _DIVS],
    "elite": [("MASTER", None), ("GRANDMASTER", None), ("CHALLENGER", None)],
}

APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}

# ------------------------------------------------------------------ player fetch

def _puuids_from_entries(entries: list[dict[str, Any]]) -> list[str]:
    """Extract PUUIDs from League-V4 entry dicts.

    Riot's API now returns ``puuid`` directly in league entries (2024+).
    Fall back to ``summonerId`` for older responses (resolved separately).
    Returns a (puuids, summoner_ids) pair.
    """
    puuids = [e["puuid"] for e in entries if e.get("puuid")]
    summoner_ids = [e["summonerId"] for e in entries if e.get("summonerId") and not e.get("puuid")]
    return puuids, summoner_ids


def sample_puuids_for_bucket(
    client: RiotClient,
    bucket: str,
    n: int,
    queue: str = "RANKED_SOLO_5x5",
) -> list[str]:
    """Return up to ``n`` PUUIDs sampled uniformly across the full division range."""
    tier_divs = list(BUCKET_ALL_DIVISIONS[bucket])
    random.shuffle(tier_divs)

    all_puuids: list[str] = []
    summoner_ids: list[str] = []
    api_errors: list[str] = []
    first_entry_keys: list[str] = []  # captured once for diagnostics

    for tier, division in tier_divs:
        if tier in APEX_TIERS:
            try:
                league = client.get_apex_league(tier, queue=queue)
                entries = league.get("entries", [])
                if entries and not first_entry_keys:
                    first_entry_keys = list(entries[0].keys())
                puuids, sids = _puuids_from_entries(entries)
                all_puuids.extend(puuids)
                summoner_ids.extend(sids)
            except Exception as exc:
                api_errors.append(f"{tier}: {exc}")
        else:
            page = random.randint(1, 2)
            try:
                entries = client.get_league_entries(tier, division, queue=queue, page=page)
                if entries and not first_entry_keys:
                    first_entry_keys = list(entries[0].keys())
                puuids, sids = _puuids_from_entries(entries)
                all_puuids.extend(puuids)
                summoner_ids.extend(sids)
            except Exception as exc:
                api_errors.append(f"{tier} {division} p{page}: {exc}")

        if len(all_puuids) + len(summoner_ids) >= n * 3:
            break

    # Resolve any old-style summonerIds to PUUIDs
    for sid in summoner_ids:
        try:
            summoner = client.get_summoner_by_id(sid)
            puuid = summoner.get("puuid", "")
            if puuid:
                all_puuids.append(puuid)
        except Exception as exc:
            print(f"  WARNING: could not resolve summonerId {sid}: {exc}")

    if not all_puuids:
        diag = ""
        if first_entry_keys:
            diag = f"\n  Entry fields returned by API: {first_entry_keys}"
        elif not api_errors:
            diag = "\n  API returned 200 but all entries were empty lists."
        error_lines = "\n  ".join(api_errors) if api_errors else "no HTTP errors"
        raise RuntimeError(
            f"[{client.config.platform}] 0 PUUIDs collected for bucket={bucket!r}.\n"
            f"  API errors: {error_lines}{diag}\n"
            f"  Check: is your RIOT_API_KEY valid and non-expired?"
        )

    random.shuffle(all_puuids)
    selected = all_puuids[:n]
    print(f"  [{client.config.platform}] {bucket}: {len(selected)} players")
    return selected


# ----------------------------------------------------------------- interleaving

@dataclass
class _RegionQueue:
    """Per-region work queue with cooldown tracking."""
    platform: str
    client: RiotClient
    puuids: list[str] = field(default_factory=list)
    cooldown_until: float = 0.0
    matches_cached: int = 0

    def is_available(self) -> bool:
        return bool(self.puuids) and time.monotonic() >= self.cooldown_until

    def mark_cooling(self, retry_after: float) -> None:
        self.cooldown_until = time.monotonic() + retry_after
        logger.warning(
            "[%s] rate-limited, cooling for %.0fs — switching to next region",
            self.platform, retry_after,
        )


def _interleaved_download(
    region_queues: list[_RegionQueue],
    matches_per_player: int,
    queue: int,
) -> dict[str, int]:
    """Drain all region queues, cycling when a region is cooling."""
    total_players = sum(len(rq.puuids) for rq in region_queues)
    done = 0

    while any(rq.puuids for rq in region_queues):
        now = time.monotonic()

        # Find an available queue
        available = [rq for rq in region_queues if rq.is_available()]
        if not available:
            # All cooling — wait for the soonest recovery
            soonest = min(
                rq.cooldown_until for rq in region_queues if rq.puuids
            )
            wait = soonest - now
            logger.info("All regions cooling, waiting %.0fs", wait)
            time.sleep(max(0.0, wait) + 0.1)
            continue

        # Round-robin: pick the queue with the longest wait since last served
        # (simple: just take the first available one, rotate after use)
        rq = available[0]
        region_queues.remove(rq)
        region_queues.append(rq)  # move to end of rotation

        puuid = rq.puuids[0]
        try:
            fetched = download_for_puuid(
                rq.client, puuid,
                count=matches_per_player,
                queue=queue,
            )
            rq.puuids.pop(0)
            rq.matches_cached += len(fetched)
            done += 1
            logger.info(
                "[%s] %d/%d players — cached %d matches (region total: %d)",
                rq.platform, done, total_players, len(fetched), rq.matches_cached,
            )
        except RateLimitError as e:
            rq.mark_cooling(e.retry_after)
            # PUUID stays at front of queue; will retry after cooldown

    return {rq.platform: rq.matches_cached for rq in region_queues}


# ----------------------------------------------------------------------- main

def seed_matches(
    platforms: list[str],
    buckets: list[str],
    *,
    players_per_tier: int = 30,
    matches_per_player: int = 20,
    queue: int = 420,
) -> dict[str, int]:
    base_cfg = load_riot_config()
    api_key = base_cfg.api_key

    region_queues: list[_RegionQueue] = []

    for platform in platforms:
        region = PLATFORM_TO_REGION.get(platform)
        if region is None:
            logger.error("Unknown platform %s, skipping", platform)
            continue

        cfg = RiotConfig(api_key=api_key, platform=platform, region=region)
        # raise_on_rate_limit=True so the interleaver can catch 429 and switch
        client = RiotClient(cfg, raise_on_rate_limit=True)

        print(f"[{platform}] fetching player lists...")
        puuids: list[str] = []
        for bucket in buckets:
            bucket_puuids = sample_puuids_for_bucket(client, bucket, players_per_tier)
            puuids.extend(bucket_puuids)
            storage.save_puuid_buckets({p: bucket for p in bucket_puuids})

        random.shuffle(puuids)
        region_queues.append(_RegionQueue(platform=platform, client=client, puuids=puuids))

    if not region_queues:
        print("No valid regions configured.")
        return {}

    total_players = sum(len(rq.puuids) for rq in region_queues)
    print(f"\nDownloading matches for {total_players} players across {len(region_queues)} region(s)...\n")

    results = _interleaved_download(region_queues, matches_per_player, queue)

    for platform, count in results.items():
        print(f"[{platform}] {count} matches cached")
    print(f"\nGrand total: {sum(results.values())} matches")
    return results


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=["na1"],
        metavar="PLATFORM",
        help=(
            "Platform routing values to collect from (default: na1). "
            "Each adds an independent rate-limit pool. "
            f"Supported: {', '.join(sorted(PLATFORM_TO_REGION))}."
        ),
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        choices=list(BUCKET_ALL_DIVISIONS),
        default=list(BUCKET_ALL_DIVISIONS),
        metavar="BUCKET",
        help="Rank buckets: low mid high elite (default: all four)",
    )
    p.add_argument(
        "--players-per-tier",
        type=int,
        default=30,
        help="Players per bucket per region (default 30)",
    )
    p.add_argument(
        "--matches-per-player",
        type=int,
        default=20,
        help="Recent ranked matches per player (default 20)",
    )
    p.add_argument(
        "--queue",
        type=int,
        default=420,
        help="Riot queue ID (default 420 = ranked solo/duo)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sampling")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously — start a new pass as soon as the previous one finishes.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _count_cached() -> int:
    """Return number of match JSON files in the raw cache."""
    from ..config import RAW_DIR
    d = RAW_DIR / "matches"
    return sum(1 for _ in d.glob("*.json")) if d.exists() else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.seed is not None:
        random.seed(args.seed)

    pass_num = 0
    while True:
        pass_num += 1
        before = _count_cached()
        t0 = time.monotonic()

        if args.loop:
            print(f"\n{'='*58}")
            print(f"  Pass {pass_num}  |  {before:,} matches cached so far")
            print(f"{'='*58}\n")

        try:
            seed_matches(
                platforms=args.regions,
                buckets=args.tiers,
                players_per_tier=args.players_per_tier,
                matches_per_player=args.matches_per_player,
                queue=args.queue,
            )
        except KeyboardInterrupt:
            after = _count_cached()
            mins = (time.monotonic() - t0) / 60
            print(f"\nStopped.  +{after - before:,} new matches this pass  |  "
                  f"{after:,} total  |  {mins:.1f} min elapsed")
            return 0

        after = _count_cached()
        mins = (time.monotonic() - t0) / 60
        print(f"\nPass {pass_num} done — +{after - before:,} new  |  "
              f"{after:,} total  |  {mins:.1f} min")

        if not args.loop:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
