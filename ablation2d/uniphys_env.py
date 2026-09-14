"""Import UniPhys from the source checkout, LAZILY.

Only the dataset generator needs the solver. Training, the metrics and the
interactive planner run on the `.npz` corpus and a baked tissue table, so
importing this module must not require UniPhys to be installed — otherwise the
notebook cannot open on Colab, which is the only place it will ever be run by
anyone but us.

That was not true in the first draft: `device.py` imported UniPhys at module
scope, `anatomy` imported `device`, `channels` imported `anatomy`, and the whole
package became unimportable without a compiled C++ solver. Hence
`require_uniphys()` rather than a module-level `uniphys`.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_CANDIDATES = [
    os.environ.get("UNIPHYS_PYTHON"),
    "/home/jonas/Documents/Research/UniPhys/python",
    str(Path.home() / "Documents/Research/UniPhys/python"),
]

#: Entry points added for this workshop (UniPhys python/bindings.cpp). Without
#: them a Python caller silently gets the uncalibrated device and inert vessels,
#: so fail loudly rather than generate a corpus that looks fine and is not the
#: physics we claim.
_REQUIRED = ["set_mw_calibration", "_set_vessel_radius"]


@lru_cache(maxsize=1)
def require_uniphys():
    """The UniPhys module, or a clear ImportError explaining what needs it."""
    mod = None
    try:
        import uniphys as mod  # noqa: F401
    except ImportError:
        for c in _CANDIDATES:
            if c and (Path(c) / "uniphys" / "__init__.py").exists():
                sys.path.insert(0, c)
                try:
                    import uniphys as mod  # noqa: F811
                except ImportError:
                    mod = None
                break
    if mod is None:
        raise ImportError(
            "UniPhys is needed for this operation (solving new ground truth). "
            "Set UNIPHYS_PYTHON to the repo's python/ directory, or install the "
            "wheel. Training, evaluation and the interactive planner do NOT need "
            "it — they run on the .npz corpus and the baked tissue table."
        )
    missing = [n for n in _REQUIRED
               if not (hasattr(mod, n) or hasattr(mod.Simulation, n))]
    if missing:
        raise ImportError(
            f"UniPhys is too old for this workshop: missing {', '.join(missing)}. "
            "Rebuild the python module (cmake --build build/py --target _uniphys)."
        )
    return mod


def uniphys_available() -> bool:
    try:
        require_uniphys()
        return True
    except ImportError:
        return False
