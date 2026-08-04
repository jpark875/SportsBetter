"""
scripts/pull_wc_data.py

Download the international football results dataset and build the
World Cup training CSV.

No API key needed — data comes from the public GitHub dataset
maintained by martj42 (updated daily during tournaments).

Output
------
  data/wc_training_features.csv

Usage
-----
  python scripts/pull_wc_data.py
  python scripts/pull_wc_data.py --from-year 1990
  python scripts/pull_wc_data.py --wc-only   # World Cup + qualifiers only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import FOOTBALL_DATA_URL, GOALSCORERS_DATA_URL
from data_pipeline.football_stats_client import build_training_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull international football data and build WC training CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", default="data/wc_training_features.csv")
    p.add_argument(
        "--from-year", type=int, default=1993,
        help="Drop matches before this year (ELO warm-up period, ~30 years recommended).",
    )
    p.add_argument(
        "--wc-only", action="store_true",
        help="Keep only World Cup, qualifiers, and continental tournaments.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = build_training_features(
        url=FOOTBALL_DATA_URL,
        goalscorers_url=GOALSCORERS_DATA_URL,
        min_year=args.from_year,
        filter_wc_and_competitive=args.wc_only,
        output_path=args.output,
    )
    print(f"\nDone. {len(df):,} rows written to {args.output}")
    print(f"Result breakdown: home win {(df['RESULT']==2).mean():.1%} | "
          f"draw {(df['RESULT']==1).mean():.1%} | "
          f"away win {(df['RESULT']==0).mean():.1%}")
