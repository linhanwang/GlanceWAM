"""Set up symlinks so torchcodec can load PyAV's bundled ffmpeg 6 libs.

Stock Ubuntu 22.04 ships ffmpeg 4.4, but torchcodec wheels link against
ffmpeg 5+. PyAV's manylinux wheel ships its own ffmpeg 6 .so files inside
``.venv/lib/python3.11/site-packages/av.libs/``, but with hash-suffixed
filenames (``libavutil-d2f6c3c3.so.58.29.100``) that the dynamic linker
won't pick up on soname lookup.

This script creates standard-named symlinks (``libavutil.so.58`` etc.) in
``.venv/lib/ffmpeg6_shim/`` pointing at the PyAV-bundled libs. Launch
scripts then prepend ``ffmpeg6_shim/`` and ``av.libs/`` to LD_LIBRARY_PATH
so torchcodec dlopens the standard-named symlinks (resolving to PyAV's
libs) and their hash-named transitive deps (libvpx, libdav1d, etc.)
resolve from av.libs/.

Run once after ``uv sync``. Idempotent — safe to re-run if PyAV is bumped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AV_LIBS = REPO_ROOT / ".venv/lib/python3.11/site-packages/av.libs"
SHIM_DIR = REPO_ROOT / ".venv/lib/ffmpeg6_shim"

SHIM_TARGETS: list[tuple[str, str]] = [
    ("libavcodec.so.60", "libavcodec-*.so.60.*"),
    ("libavformat.so.60", "libavformat-*.so.60.*"),
    ("libavutil.so.58", "libavutil-*.so.58.*"),
    ("libavfilter.so.9", "libavfilter-*.so.9.*"),
    ("libavdevice.so.60", "libavdevice-*.so.60.*"),
    ("libswresample.so.4", "libswresample-*.so.4.*"),
    ("libswscale.so.7", "libswscale-*.so.7.*"),
    ("libpostproc.so.57", "libpostproc-*.so.57.*"),
]


def main() -> int:
    if not AV_LIBS.is_dir():
        print(f"error: {AV_LIBS} not found - install av (uv sync)", file=sys.stderr)
        return 1
    SHIM_DIR.mkdir(parents=True, exist_ok=True)
    for soname, pattern in SHIM_TARGETS:
        matches = sorted(AV_LIBS.glob(pattern))
        if not matches:
            print(f"error: no PyAV lib matched {pattern} in {AV_LIBS}", file=sys.stderr)
            return 1
        if len(matches) > 1:
            print(f"warning: multiple matches for {pattern}: {matches}", file=sys.stderr)
        target = os.path.relpath(matches[0], SHIM_DIR)
        link = SHIM_DIR / soname
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    print(f"ffmpeg6 shim ready at {SHIM_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
