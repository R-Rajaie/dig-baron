"""Download Match-V5 match + timeline JSON to the local cache.

Usage:
    python -m lolobj.ingest.download_matches \
        --riot-id "SummonerName#TAG" \
        --count 20 --queue 420

Queue ids:
    420  Ranked Solo/Duo
    440  Ranked Flex
    400  Normal Draft
    430  Normal Blind
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from ..riot_client import RiotClient
from . import storage

logger = logging.getLogger(__name__)


def download_for_puuid(
    client: RiotClient,
    puuid: str,
    *,
    count: int = 20,
    queue: int | None = 420,
    match_type: str | None = None,
    skip_cached: bool = True,
) -> list[str]:
    """Fetch ``count`` match_ids for ``puuid`` and cache both match and timeline JSON.

    Returns the list of match_ids successfully cached (newly downloaded or already present).
    """
    fetched: list[str] = []
    # Riot caps count at 100 per request; loop in chunks of 100.
    remaining = count
    start = 0
    while remaining > 0:
        chunk = min(remaining, 100)
        ids = client.get_match_ids_by_puuid(
            puuid, start=start, count=chunk, queue=queue, match_type=match_type
        )
        if not ids:
            break
        for match_id in ids:
            try:
                _ensure_cached(client, match_id, skip_cached=skip_cached)
                fetched.append(match_id)
            except Exception:  # network / 4xx / 5xx — log and continue
                logger.exception("Failed to cache %s", match_id)
        start += len(ids)
        remaining -= len(ids)
        if len(ids) < chunk:
            break
    return fetched


def _ensure_cached(client: RiotClient, match_id: str, *, skip_cached: bool) -> None:
    if not (skip_cached and storage.has_match(match_id)):
        storage.save_match(match_id, client.get_match(match_id))
    if not (skip_cached and storage.has_timeline(match_id)):
        storage.save_timeline(match_id, client.get_match_timeline(match_id))


def download_for_puuids(
    client: RiotClient,
    puuids: Iterable[str],
    *,
    count: int = 20,
    queue: int | None = 420,
) -> list[str]:
    out: list[str] = []
    for puuid in puuids:
        out.extend(download_for_puuid(client, puuid, count=count, queue=queue))
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--riot-id", help='Riot ID in the form "Name#TAG"')
    g.add_argument("--puuid", help="Account PUUID")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--queue", type=int, default=420, help="Riot queue id (default 420 = ranked solo)")
    p.add_argument("--type", dest="match_type", default=None, help="Match type filter, e.g. ranked")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    client = RiotClient()
    if args.riot_id:
        if "#" not in args.riot_id:
            print('--riot-id must be in the form "Name#TAG"', file=sys.stderr)
            return 2
        game_name, tag_line = args.riot_id.split("#", 1)
        account = client.get_account_by_riot_id(game_name, tag_line)
        puuid = account["puuid"]
    else:
        puuid = args.puuid
    fetched = download_for_puuid(
        client, puuid, count=args.count, queue=args.queue, match_type=args.match_type
    )
    print(f"Cached {len(fetched)} matches for puuid={puuid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
