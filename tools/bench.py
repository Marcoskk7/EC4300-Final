"""Materialise the runnable benchmark: pristine upstream + our patch, under .build/.

`third_party/HypoSpace/` must stay byte-identical to upstream commit c69e931 so the repo diff
only shows our own work. Anything that imports the benchmark calls `ensure(py, root)`; the
result is `.build/hypospace/`, which is gitignored and rebuilt when the patch mtime is newer.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/HypoSpace"
BUILD = ROOT / ".build/hypospace"
PATCH = ROOT / "patches/hypospace-local-llm.patch"
STAMP = ROOT / ".build/.patch-stamp"


def ensure():
    """Copy upstream to .build/ and apply the patch once. Idempotent."""
    fresh = not BUILD.exists()
    if fresh:
        shutil.copytree(UPSTREAM, BUILD, ignore=shutil.ignore_patterns("__pycache__"))

    stamp = STAMP.read_text().strip() if STAMP.exists() else ""
    want = f"{PATCH.stat().st_mtime_ns}"
    if not fresh and stamp == want:
        return BUILD

    if not fresh:  # upstream or patch changed: rebuild from scratch
        shutil.rmtree(BUILD)
        shutil.copytree(UPSTREAM, BUILD, ignore=shutil.ignore_patterns("__pycache__"))

    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(PATCH)], cwd=BUILD, capture_output=True, text=True
    )
    if r.returncode != 0:
        raise SystemExit(f"failed to apply {PATCH.name} to {BUILD}:\n{r.stderr}")
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(want)
    return BUILD


if __name__ == "__main__":
    print(ensure())
