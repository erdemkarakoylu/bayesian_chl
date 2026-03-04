from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib.patches import Patch


PPC_COLORS = ['darkgray', 'k', 'C1']

# ============================================================================
# CRITICAL: These must match plot_style.py EXACTLY
# ============================================================================
FONT_SIZES = {
    'title': 10,
    'label': 9,
    'tick': 8,
    'legend': 8,
}


def _force_font_sizes(ax):
    """
    FORCE font sizes after ArviZ plotting.
    ArviZ overrides rcParams, so we must re-apply sizes AFTER plotting.
    """
    # Title
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'])
    
    # Axis labels
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    
    # Legend
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(FONT_SIZES['legend'])
        title_obj = leg.get_title()
        if title_obj:
            title_obj.set_fontsize(FONT_SIZES['legend'])


def plot_ppc_only(
    idata,
    *,
    y: str = "likelihood",
    colors=None,
    figsize: Tuple[float, float] = (8, 4.5),
    ax: Optional[pp.Axes] = None,
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Posterior Predictive Check (PPC) plot for a single model.
    Font sizes are enforced AFTER ArviZ plotting.
    """
    if ax is None:
        fig, ax = pp.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ArviZ will override fonts - we fix it after
    az.plot_ppc(idata, data_pairs={y: y}, colors=colors, ax=ax)
    ax.set_title("Posterior predictive check")
    
    # CRITICAL: Force font sizes after ArviZ
    _force_font_sizes(ax)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_loo_pit_ecdf_compare(
    idata5,
    idata1,
    *,
    y: str = "likelihood",
    figsize: Tuple[float, float] = (8, 4.5),
    ax: Optional[pp.Axes] = None,
    model5_label: str = "Model 5",
    model1_label: str = "Model 1",
    model1_style: str = "--",
    model1_color: str = "0.6",
    envelope_label: str = "94% reference envelope for uniform LOO-PIT",
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Plot Model 5 ECDF residual first, then call ArviZ LOO-PIT ECDF for Model 1.
    Font sizes are enforced AFTER ArviZ plotting.
    """

    def _to_1d_array(pit_obj) -> np.ndarray:
        pit = np.asarray(pit_obj).ravel()
        pit = pit[np.isfinite(pit)]
        pit = np.clip(pit, 0.0, 1.0)
        return pit

    def _ecdf(x: np.ndarray):
        x = np.sort(x)
        n = x.size
        F = np.arange(1, n + 1) / n
        return x, F

    if ax is None:
        fig, ax = pp.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- 1) Plot Model 5 ECDF residual ---
    pit5 = _to_1d_array(az.loo_pit(idata5, y=y))
    u5, F5 = _ecdf(pit5)
    (h5,) = ax.plot(u5, u5 - F5, label=model5_label, linewidth=1.5)

    # --- 2) Overlay ArviZ LOO-PIT ECDF for Model 1 ---
    az.plot_loo_pit(idata1, y=y, ax=ax, ecdf=True, legend=False)

    # Find and style the Model 1 line
    new_lines = ax.lines[-3:]
    model1_line = None
    for ln in reversed(new_lines):
        yd = np.asarray(ln.get_ydata())
        if yd.size and not np.allclose(yd, yd[0]):
            model1_line = ln
            break
    if model1_line is not None:
        model1_line.set_linestyle(model1_style)
        model1_line.set_color(model1_color)
        model1_line.set_linewidth(1.5)
        model1_line.set_label(model1_label)
        h1 = model1_line
    else:
        (h1,) = ax.plot([], [], linestyle=model1_style, color=model1_color, 
                       linewidth=1.5, label=model1_label)

    # --- 3) Legend ---
    envelope_patch = Patch(alpha=0.15, label=envelope_label)
    ax.legend(handles=[h5, h1, envelope_patch], frameon=False)

    # --- 4) Labels ---
    ax.set_xlabel(r"$u = p(y_i \mid y_{-i})$")
    ax.set_ylabel(r"$u - \widehat{F}_{\mathrm{PIT}}(u)$")
    ax.set_title("LOO-PIT ECDF residual (closer to 0 indicates better calibration)")
    ax.set_xlim(0.0, 1.0)
    
    # CRITICAL: Force font sizes after ArviZ
    _force_font_sizes(ax)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def _ppc_hdi_from_pp(
    y_pp: np.ndarray,
    hdi_prob: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median, hdi_low, hdi_high across draws for each observation index."""
    y_pp = np.asarray(y_pp)
    if y_pp.ndim != 2:
        raise ValueError(f"Expected y_pp with shape (n_draws, n_obs); got {y_pp.shape}")
    qlo = (1.0 - hdi_prob) / 2.0
    qhi = 1.0 - qlo
    med = np.nanquantile(y_pp, 0.5, axis=0)
    lo = np.nanquantile(y_pp, qlo, axis=0)
    hi = np.nanquantile(y_pp, qhi, axis=0)
    return med, lo, hi


def _sample_pp_for_dataframe(
    *,
    model: pm.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    y_name: str = "likelihood",
    x_data_name: str = "log(MBR)",
    y_data_name: str = "log(chl)",
    group_name: str = "group_idx",
    group_flag_col: str = "MBR_flag",
    coords_name_obs: str = "obs_idx",
    seed: int = 0,
) -> np.ndarray:
    """Sample posterior predictive for df using an existing fitted model+idata.posterior."""
    coords = {coords_name_obs: df.index}

    set_data = {
        x_data_name: df["log_MBR"].to_numpy(),
        y_data_name: df["log_chl"].to_numpy(),
    }

    with model:
        if group_name in model.named_vars:
            group_idx, _ = pd.factorize(df[group_flag_col], sort=True)
            set_data[group_name] = group_idx

        pm.set_data(set_data, coords=coords)

        ppc = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=[y_name],
            random_seed=seed,
            progressbar=False,
        )

    arr = np.asarray(ppc.posterior_predictive[y_name])
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected PPC array shape for {y_name}: {arr.shape}")

    return arr


def plot_regression_hdi_2panel(
    *,
    idata: az.InferenceData,
    model: pm.Model,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    hdi_prob: float = 0.94,
    y_name: str = "likelihood",
    seed: int = 0,
    show_points: bool = True,
    point_alpha: float = 0.45,
    title_left: str = "In-sample (NOMAD)",
    title_right: str = "Out-of-sample (SeaBASS)",
) -> Tuple[pp.Figure, np.ndarray]:
    """
    Two-panel predictive coverage plot.
    Font sizes are applied explicitly (pure matplotlib, no ArviZ).
    """
    fig, axes = pp.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # ----- In-sample -----
    ypp_tr = _sample_pp_for_dataframe(
        model=model, idata=idata, df=df_train, y_name=y_name, seed=seed
    )
    med_tr, lo_tr, hi_tr = _ppc_hdi_from_pp(ypp_tr, hdi_prob=hdi_prob)

    x_tr = df_train["log_MBR"].to_numpy()
    order_tr = np.argsort(x_tr)
    axes[0].fill_between(x_tr[order_tr], lo_tr[order_tr], hi_tr[order_tr], 
                         alpha=0.25, linewidth=0)
    axes[0].plot(x_tr[order_tr], med_tr[order_tr], linewidth=1.5)

    if show_points:
        axes[0].scatter(df_train["log_MBR"], df_train["log_chl"],
                       alpha=point_alpha, s=12, edgecolors="none", zorder=2)

    axes[0].set_xlabel(r"$\log(\mathrm{MBR})$", fontsize=FONT_SIZES['label'])
    axes[0].set_ylabel(r"$\log_{10}(\mathrm{Chl\!-\!a})$", fontsize=FONT_SIZES['label'])
    axes[0].set_title(title_left, fontsize=FONT_SIZES['title'])
    axes[0].tick_params(axis='both', labelsize=FONT_SIZES['tick'])

    # ----- Out-of-sample -----
    ypp_te = _sample_pp_for_dataframe(
        model=model, idata=idata, df=df_test, y_name=y_name, seed=seed + 1
    )
    med_te, lo_te, hi_te = _ppc_hdi_from_pp(ypp_te, hdi_prob=hdi_prob)

    x_te = df_test["log_MBR"].to_numpy()
    order_te = np.argsort(x_te)
    axes[1].fill_between(x_te[order_te], lo_te[order_te], hi_te[order_te], 
                         alpha=0.25, linewidth=0)
    axes[1].plot(x_te[order_te], med_te[order_te], linewidth=1.5)

    if show_points:
        axes[1].scatter(df_test["log_MBR"], df_test["log_chl"],
                       alpha=point_alpha, s=12, edgecolors="none", zorder=2)

    axes[1].set_xlabel(r"$\log(\mathrm{MBR})$", fontsize=FONT_SIZES['label'])
    axes[1].set_title(title_right, fontsize=FONT_SIZES['title'])
    axes[1].tick_params(axis='both', labelsize=FONT_SIZES['tick'])

    fig.tight_layout()
    return fig, axes
