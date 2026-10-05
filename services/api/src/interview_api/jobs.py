"""Scheduled jobs: ``python -m interview_api.jobs purge``."""

from __future__ import annotations

import json
import sys

from interview_api.db import get_db
from interview_api.routers.privacy import purge_expired


def main(argv: list[str]) -> int:
    if argv[:1] != ["purge"]:
        print("usage: python -m interview_api.jobs purge", file=sys.stderr)
        return 2
    db = next(get_db())
    print(json.dumps(purge_expired(db)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
