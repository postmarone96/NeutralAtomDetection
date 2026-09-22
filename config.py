import os
from pathlib import Path

matplotlib_cache_dir = Path("/tmp") / "neutral_atom_matplotlib"
font_cache_dir = Path("/tmp") / "neutral_atom_cache"


def use_headless_matplotlib():
    """Point matplotlib's caches somewhere writable.

    Must run before `import matplotlib` -- MPLCONFIGDIR is read at import time,
    so setting it afterwards has no effect.
    """
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    font_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_dir))
