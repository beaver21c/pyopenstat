"""Download Mplus User's Guide example pages (input + Mplus 8.8 output) and data files from statmodel.com.

The files are used by tests/test_sem_mplus_examples.py to check the SEM engine against published Mplus output.
They are not redistributed with this repository (copyright Muthén & Muthén); this script fetches them into
tests/data/mplus_ug/ (ignored by git).

    python scripts/fetch_mplus_examples.py
"""
from __future__ import annotations

import os
import sys
import urllib.request

EXAMPLES = ["3.11", "3.12", "3.13", "3.14", "5.1", "5.2", "5.3", "5.6", "5.8", "5.9", "5.10", "5.11", "5.12",
            "6.1", "6.4", "6.5", "6.9", "6.10", "6.11", "6.13", "6.14", "6.15"]
BASE = "https://www.statmodel.com/usersguide"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data", "mplus_ug")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ok = 0
    for ex in EXAMPLES:
        chap = ex.split(".")[0]
        for ext in ("html", "dat"):
            url = f"{BASE}/chap{chap}/ex{ex}.{ext}"
            dest = os.path.join(OUT, f"ex{ex}.{ext}")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                ok += 1
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "openstat-test-fetch/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                    f.write(r.read())
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"failed: {url} ({e})", file=sys.stderr)
    print(f"{ok}/{2 * len(EXAMPLES)} files in {OUT}")
    return 0 if ok == 2 * len(EXAMPLES) else 1


if __name__ == "__main__":
    sys.exit(main())
