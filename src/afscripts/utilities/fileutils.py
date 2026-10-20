"""
File and filesystem utility functions for Hilbert workflows.
"""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def chdir(target_dir, create=False):
    """
    Context manager to temporarily change the working directory.

    :param target_dir: Target directory path to switch into.
    :param create: If True, create the directory if it does not exist.
    :return: Yields resolved Path object to the target directory.
    """
    prev_dir = Path.cwd()
    target = Path(target_dir).resolve()

    if create and not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(prev_dir)


def find_upstream_files(pattern: str, root_dir: Path | str = "upstream") -> list[Path]:
    """
    Search directory tree following symlinks for files matching glob pattern.

    :param pattern: Glob pattern to match filenames against.
    :type pattern: str
    :param root_dir: Root directory path to search within.
    :type root_dir: Path | str
    :return: List of unique matching Path objects.
    :rtype: list[Path]
    """
    root = Path(root_dir)
    if not root.is_dir():
        return []
    matched: list[Path] = []
    for dirpath, _, _ in os.walk(root, followlinks=True):
        matched.extend(list(Path(dirpath).glob(pattern)))
    return list(dict.fromkeys(matched))

