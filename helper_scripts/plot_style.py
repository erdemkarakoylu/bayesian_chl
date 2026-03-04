# helper_scripts/plot_style.py
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import matplotlib as mpl
import matplotlib.pyplot as pp


# One place to edit global style
PLOT_STYLE: Dict[str, Any] = {
    # ---- Figure / saving defaults ----
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # ---- Fonts ----
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # ---- Axes / grid ----
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,

    # ---- Lines ----
    "lines.linewidth": 2.0,
    "lines.markersize": 4.5,

    # ---- Legend ----
    "legend.frameon": False,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.3,

    # ---- Math text ----
    "mathtext.default": "it",

    # ---- Layout ----
    "figure.constrained_layout.use": False,  # you control via tight_layout()
}


def set_plot_style(extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply the project-wide matplotlib style globally.

    Call this once near the top of each notebook/script, ideally before plotting.
    If you also use ArviZ styles, call this AFTER az.style.use(...).
    """
    mpl.rcParams.update(PLOT_STYLE)
    if extra:
        mpl.rcParams.update(extra)


@contextmanager
def plot_context(extra: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """
    Temporary style override for a single plot or block of plots.

    Example:
        with plot_context({"font.size": 12}):
            ...
    """
    old = mpl.rcParams.copy()
    try:
        set_plot_style(extra=extra)
        yield
    finally:
        mpl.rcParams.update(old)


def savefig(fig: pp.Figure, path: Path, *, dpi: Optional[int] = None) -> None:
    """
    Consistent save wrapper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi if dpi is not None else mpl.rcParams["savefig.dpi"])
