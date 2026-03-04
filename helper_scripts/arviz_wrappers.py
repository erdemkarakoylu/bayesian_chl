"""
Plot wrappers for ArviZ with automatic font standardization.

These functions wrap ArviZ plotting functions and automatically enforce
consistent font sizes across all figures. Use these instead of calling
ArviZ directly to ensure standardized styling.
"""

from pathlib import Path
from typing import Optional, List, Union
import matplotlib.pyplot as pp
import matplotlib as mpl
import arviz as az
import numpy as np
try:
    import graphviz
    from PIL import Image
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# ============================================================================
# FONT SIZE CONSTANTS (match plot_style.py)
# ============================================================================

FONT_SIZES = {
    'title': 10,
    'label': 9,
    'tick': 8,
    'legend': 8,
}


def _force_font_sizes(ax):
    """
    Internal helper: Force consistent font sizes on axes.
    This is needed because ArviZ overrides matplotlib rcParams.
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


# ============================================================================
# PUBLIC WRAPPER FUNCTIONS
# ============================================================================

def plot_trace_standardized(
    idata,
    var_names: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    compact: bool = False,
    **kwargs
):
    """
    Trace plot with standardized fonts.
    
    Wrapper around az.plot_trace that automatically enforces consistent
    font sizes across all subplots.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted model inference data
    var_names : list of str, optional
        Variables to plot. If None, plots all.
    figsize : tuple
        Figure size (width, height) in inches
    compact : bool
        Use compact layout (default False)
    **kwargs
        Additional arguments passed to az.plot_trace
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    axes : numpy.ndarray
        Array of axes objects
        
    Example
    -------
    >>> fig, axes = plot_trace_standardized(idata1, figsize=(12, 8))
    >>> fig.savefig('model1_traceplot.jpeg', dpi=300, format='jpeg')
    """
    axes = az.plot_trace(idata, var_names=var_names, compact=compact, 
                         figsize=figsize, **kwargs)
    
    # Enforce fonts on all axes
    for ax in axes.flatten():
        _force_font_sizes(ax)
    
    fig = pp.gcf()
    return fig, axes


def plot_forest_standardized(
    idata,
    var_names: Optional[List[str]] = None,
    combined: bool = True,
    hdi_prob: float = 0.94,
    figsize: tuple = (8, 6),
    **kwargs
):
    """
    Forest plot with standardized fonts.
    
    Wrapper around az.plot_forest that automatically enforces consistent
    font sizes.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted model inference data
    var_names : list of str, optional
        Variables to plot
    combined : bool
        Combine chains (default True)
    hdi_prob : float
        HDI probability (default 0.94)
    figsize : tuple
        Figure size (width, height) in inches
    **kwargs
        Additional arguments passed to az.plot_forest
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> fig, ax = plot_forest_standardized(idata1, var_names=['alpha', 'beta'])
    >>> fig.savefig('model1_forest.jpeg', dpi=300, format='jpeg')
    """
    axes = az.plot_forest(idata, var_names=var_names, combined=combined,
                       hdi_prob=hdi_prob, figsize=figsize, **kwargs)
    
    # Enforce fonts
    # Handle both single Axes and array of Axes
    if isinstance(axes, np.ndarray):
        # Array of axes - apply to each
        for ax in axes.flatten():
            _force_font_sizes(ax)
    else:
        # Single axes
        _force_font_sizes(axes)
    
    fig = pp.gcf()
    return fig, axes


def plot_ppc_standardized(
    idata,
    group: str = 'posterior',
    kind: str = 'kde',
    data_pairs: Optional[dict] = None,
    colors: Optional[list] = None,
    figsize: tuple = (8, 5),
    ax: Optional[pp.Axes] = None,
    **kwargs
):
    """
    Posterior/prior predictive check with standardized fonts.
    
    Wrapper around az.plot_ppc that automatically enforces consistent
    font sizes.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted model inference data
    group : str
        'posterior' or 'prior' (default 'posterior')
    kind : str
        'kde' or 'cumulative' (default 'kde')
    data_pairs : dict, optional
        Data pairs for comparison
    colors : list, optional
        Color list (e.g., ['darkgray', 'k', 'C1'])
    figsize : tuple
        Figure size if ax is None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to az.plot_ppc
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> fig, ax = plot_ppc_standardized(idata1, group='prior', kind='cumulative')
    >>> ax.set_xlabel('log(Chl-a)')
    >>> fig.savefig('model1_ppc.jpeg', dpi=300, format='jpeg')
    """
    if ax is None:
        fig, ax = pp.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    az.plot_ppc(idata, group=group, kind=kind, data_pairs=data_pairs,
               colors=colors, ax=ax, **kwargs)
    
    # Enforce fonts
    _force_font_sizes(ax)
    
    return fig, ax


def plot_loo_pit_standardized(
    idata,
    y: Optional[str] = None,
    ecdf: bool = True,
    figsize: tuple = (8, 5),
    ax: Optional[pp.Axes] = None,
    **kwargs
):
    """
    LOO-PIT diagnostic plot with standardized fonts.
    
    Wrapper around az.plot_loo_pit that automatically enforces consistent
    font sizes.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted model inference data
    y : str, optional
        Variable name (if None, uses 'y')
    ecdf : bool
        Use ECDF plot (default True)
    figsize : tuple
        Figure size if ax is None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to az.plot_loo_pit
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> fig, ax = plot_loo_pit_standardized(idata1, y='likelihood', ecdf=True)
    >>> fig.savefig('model1_loo_pit.jpeg', dpi=300, format='jpeg')
    """
    if ax is None:
        fig, ax = pp.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    az.plot_loo_pit(idata, y=y, ecdf=ecdf, ax=ax, **kwargs)
    
    # Enforce fonts
    _force_font_sizes(ax)
    
    return fig, ax


def plot_compare_standardized(
    comp_df,
    insample_dev: bool = True,
    figsize: tuple = (10, 6),
    **kwargs
):
    """
    Model comparison plot with standardized fonts.
    
    Wrapper around az.plot_compare that automatically enforces consistent
    font sizes.
    
    Parameters
    ----------
    comp_df : pandas.DataFrame
        Result from az.compare()
    insample_dev : bool
        Show in-sample deviance (default True)
    figsize : tuple
        Figure size
    **kwargs
        Additional arguments passed to az.plot_compare
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> comp = az.compare({'M1': idata1, 'M2': idata2})
    >>> fig, ax = plot_compare_standardized(comp, insample_dev=True)
    >>> fig.savefig('model_comparison.jpeg', dpi=300, format='jpeg')
    """
    ax = az.plot_compare(comp_df, insample_dev=insample_dev, figsize=figsize, **kwargs)
    
    # Enforce fonts
    _force_font_sizes(ax)
    
    fig = pp.gcf()
    return fig, ax


# ============================================================================
# CONVENIENCE SAVE FUNCTION
# ============================================================================

def plot_loo_pit_all_models(
    idata_dict,
    y: str = 'likelihood',
    figsize: tuple = (8, 5),
    colors: Optional[list] = None,
    linestyles: Optional[list] = None,
    **kwargs
):
    """
    LOO-PIT ECDF plot for multiple models on one axis.
    
    Shows calibration progression across models. Useful for comparing
    5+ models to see which is best calibrated.
    
    Parameters
    ----------
    idata_dict : dict
        Dictionary mapping model names to InferenceData objects
        Example: {'Model 1': idata1, 'Model 2': idata2, ...}
    y : str
        Observed variable name (default 'likelihood')
    figsize : tuple
        Figure size (width, height) in inches
    colors : list of str, optional
        Colors for each model. If None, uses colorblind-friendly palette.
    linestyles : list of str, optional
        Line styles for each model. If None, uses solid lines.
    **kwargs
        Additional arguments passed to az.plot_loo_pit
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> idata_dict = {
    ...     'Model 1': idata1,
    ...     'Model 2': idata2,
    ...     'Model 3': idata3,
    ...     'Model 4': idata4,
    ...     'Model 5': idata5
    ... }
    >>> fig, ax = plot_loo_pit_all_models(idata_dict, y='likelihood')
    >>> ax.set_title('Calibration improves from Model 1 → 5')
    >>> save_fig_jpeg(fig, 'loo_pit_all_models')
    """
    # Default colors: colorblind-friendly progression
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    if linestyles is None:
        linestyles = ['-'] * len(idata_dict)
    
    # Ensure we have enough colors/styles
    n_models = len(idata_dict)
    if len(colors) < n_models:
        colors = colors * (n_models // len(colors) + 1)
    if len(linestyles) < n_models:
        linestyles = linestyles * (n_models // len(linestyles) + 1)
    
    fig, ax = pp.subplots(figsize=figsize)
    
    # Plot each model
    for (model_name, idata), color, linestyle in zip(idata_dict.items(), colors, linestyles):
        # Compute LOO-PIT
        pit = az.loo_pit(idata, y=y)
        pit_array = np.asarray(pit).ravel()
        pit_array = pit_array[np.isfinite(pit_array)]
        pit_array = np.clip(pit_array, 0.0, 1.0)
        
        # Compute ECDF
        u = np.sort(pit_array)
        n = u.size
        F = np.arange(1, n + 1) / n
        
        # Plot ECDF residual (u - F)
        ax.plot(u, u - F, label=model_name, color=color, 
               linestyle=linestyle, linewidth=1.5)
    
    # Add reference envelope using ArviZ helper
    # Plot invisible LOO-PIT to get envelope
    az.plot_loo_pit(list(idata_dict.values())[0], y=y, ax=ax, 
                   ecdf=True, legend=False, color='none')
    
    # Style the envelope
    for collection in ax.collections:
        collection.set_alpha(0.15)
    
    # Remove the invisible line ArviZ added
    if len(ax.lines) > n_models:
        ax.lines[-1].remove()
    
    # Labels and styling
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.set_xlabel(r"$u = p(y_i \mid y_{-i})$", fontsize=FONT_SIZES['label'])
    ax.set_ylabel(r"$u - \widehat{F}_{\mathrm{PIT}}(u)$", fontsize=FONT_SIZES['label'])
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
    
    # Legend
    ax.legend(frameon=False, fontsize=FONT_SIZES['legend'])
    
    # Enforce fonts
    _force_font_sizes(ax)
    
    fig.tight_layout()
    return fig, ax


def plot_loo_pit_comparison_2models(
    idata1,
    idata2,
    y: str = 'likelihood',
    model1_label: str = 'Model 1',
    model2_label: str = 'Model 5',
    figsize: tuple = (8, 5),
    **kwargs
):
    """
    LOO-PIT ECDF comparison for exactly 2 models.
    
    Simplified interface for main text figures comparing baseline vs best model.
    
    Parameters
    ----------
    idata1 : arviz.InferenceData
        First model (typically baseline)
    idata2 : arviz.InferenceData
        Second model (typically best model)
    y : str
        Observed variable name (default 'likelihood')
    model1_label : str
        Label for first model
    model2_label : str
        Label for second model
    figsize : tuple
        Figure size
    **kwargs
        Additional arguments
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    ax : matplotlib.axes.Axes
        The axes object
        
    Example
    -------
    >>> fig, ax = plot_loo_pit_comparison_2models(
    ...     idata1, idata5,
    ...     model1_label='Model 1 (Baseline)',
    ...     model2_label='Model 5 (Best)'
    ... )
    >>> save_fig_jpeg(fig, 'loo_pit_comparison_main')
    """
    idata_dict = {
        model1_label: idata1,
        model2_label: idata2
    }
    
    # Use contrasting colors for 2-model comparison
    colors = ['#999999', '#1f77b4']  # Gray for baseline, blue for best
    linestyles = ['--', '-']  # Dashed for baseline, solid for best
    
    fig, ax = plot_loo_pit_all_models(
        idata_dict,
        y=y,
        figsize=figsize,
        colors=colors,
        linestyles=linestyles,
        **kwargs
    )
    
    return fig, ax


def plot_ppc_comparison_2models(
    idata1,
    idata2,
    y: str = 'likelihood',
    model1_label: str = 'Model 1',
    model2_label: str = 'Model 5',
    colors: Optional[list] = None,
    figsize: tuple = (8, 8),
    **kwargs
):
    """
    Stacked posterior predictive checks for 2 models.
    
    Creates a 2-panel figure (one model per row) for direct visual comparison.
    
    Parameters
    ----------
    idata1 : arviz.InferenceData
        First model (typically baseline)
    idata2 : arviz.InferenceData
        Second model (typically best model)
    y : str
        Observed variable name (default 'likelihood')
    model1_label : str
        Label for first model
    model2_label : str
        Label for second model
    colors : list, optional
        Color scheme (default: ['darkgray', 'k', 'C1'])
    figsize : tuple
        Figure size (width, height) in inches
    **kwargs
        Additional arguments passed to az.plot_ppc
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (for saving)
    axes : numpy.ndarray
        Array of 2 axes objects
        
    Example
    -------
    >>> fig, axes = plot_ppc_comparison_2models(
    ...     idata1, idata5,
    ...     model1_label='Model 1 (Polynomial)',
    ...     model2_label='Model 5 (Hierarchical)'
    ... )
    >>> save_fig_jpeg(fig, 'ppc_comparison_main')
    """
    if colors is None:
        colors = ['darkgray', 'k', 'C1']
    
    fig, axes = pp.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Model 1 (top)
    az.plot_ppc(idata1, data_pairs={y: y}, colors=colors, ax=axes[0])
    axes[0].set_title(f"Posterior predictive check: {model1_label}", 
                     fontsize=FONT_SIZES['title'])
    _force_font_sizes(axes[0])
    
    # Model 2 (bottom)
    az.plot_ppc(idata2, data_pairs={y: y}, colors=colors, ax=axes[1])
    axes[1].set_title(f"Posterior predictive check: {model2_label}",
                     fontsize=FONT_SIZES['title'])
    _force_font_sizes(axes[1])
    
    fig.tight_layout()
    return fig, axes


def save_fig_jpeg(fig, filepath, dpi=300):
    """
    Save figure as JPEG with consistent settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str or Path
        Output path (will add .jpeg if no extension)
    dpi : int
        Resolution (default 300)
    
    Example
    -------
    >>> fig, ax = plot_trace_standardized(idata1)
    >>> save_fig_jpeg(fig, 'figures/model1_trace')  # Saves as model1_trace.jpeg
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.jpeg')
    elif filepath.suffix != '.jpeg':
        filepath = filepath.with_suffix('.jpeg')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, format='jpeg', dpi=dpi, bbox_inches='tight')
    
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"✓ Saved: {filepath.name} ({size_mb:.2f} MB)")
    
    return filepath


def render_dag_jpeg(
    dot: 'graphviz.Digraph',
    filepath: Union[str, Path],
    dpi: int = 300,
    cleanup: bool = True
):
    """
    Render GraphViz DAG directly to JPEG at 300 DPI.
    
    This is a standardized wrapper for saving model DAG visualizations
    in the same format as other figures (JPEG, 300 DPI).
    
    Parameters
    ----------
    dot : graphviz.Digraph
        The GraphViz graph object to render
    filepath : str or Path
        Output path (extension will be changed to .jpeg)
    dpi : int
        Resolution (default 300)
    cleanup : bool
        Remove intermediate PNG file (default True)
    
    Returns
    -------
    Path
        Path to saved JPEG file
        
    Example
    -------
    >>> import graphviz
    >>> dot = graphviz.Digraph(comment='Model 2')
    >>> dot.attr(rankdir='TB')
    >>> dot.node('alpha', 'α ~ N(0, 1)')
    >>> dot.node('y', 'y ~ N(μ, σ)')
    >>> dot.edge('alpha', 'y')
    >>> render_dag_jpeg(dot, 'figures/model2_dag')
    
    Notes
    -----
    GraphViz doesn't natively support JPEG output, so this function:
    1. Renders to PNG at specified DPI
    2. Converts PNG to JPEG using PIL
    3. Optionally cleans up intermediate PNG
    
    This ensures DAG figures match the format of all other figures
    (JPEG, 300 DPI) for consistency.
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError(
            "GraphViz and/or PIL not available. "
            "Install with: pip install graphviz pillow"
        )
    
    filepath = Path(filepath)
    
    # Remove extension if present
    if filepath.suffix:
        filepath = filepath.with_suffix('')
    
    # Set DPI in graph attributes
    dot.graph_attr['dpi'] = str(dpi)
    
    # Render to PNG first (GraphViz doesn't do JPEG directly)
    png_path = filepath.with_suffix('.png')
    dot.render(str(filepath), format='png', cleanup=cleanup)
    
    # Convert PNG to JPEG
    img = Image.open(png_path)
    
    # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
    if img.mode == 'RGBA':
        # Create white background
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        # Paste using alpha channel as mask
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as JPEG
    jpeg_path = filepath.with_suffix('.jpeg')
    img.save(jpeg_path, format='JPEG', dpi=(dpi, dpi), quality=95)
    
    # Clean up PNG if requested
    if cleanup and png_path.exists():
        png_path.unlink()
    
    size_mb = jpeg_path.stat().st_size / 1024 / 1024
    print(f"✓ Saved: {jpeg_path.name} ({size_mb:.2f} MB)")
    
    return jpeg_path
