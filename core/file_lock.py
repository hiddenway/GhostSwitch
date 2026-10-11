"""An exclusive lock on a file, held across processes, on macOS, Linux and Windows."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def exclusive(path: Path) -> Iterator[None]:
    """Wait until no other process holds the lock at path, then hold it inside the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            # Windows has no flock; lock the first byte, which may lie past the end of the file.
            handle.seek(0)
            while True:
                try:
                    # LK_LOCK retries for about ten seconds before it raises; another process
                    # may convert an adapter for minutes, so keep waiting.
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
