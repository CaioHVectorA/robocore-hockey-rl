"""Atomic files with bounded retries for Windows readers and virus scanners."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import os
import time


def sharing_violation(error):
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33)


@contextmanager
def atomic_destination(path, attempts=7):
    """Keep the previous complete file until replacement succeeds; clean up on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        for attempt in range(attempts):
            try:
                temp.replace(path)
                break
            except OSError as exc:
                if not sharing_violation(exc) or attempt == attempts - 1:
                    raise
                time.sleep(min(0.005 * 2**attempt, 0.04))
    finally:
        temp.unlink(missing_ok=True)
