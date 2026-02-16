"""
Forecast Academy
================

Core modules for the Real-World Forecasting Foundations course.

Structure:
- loaders: M5 data loading, preprocessing, messification
- cache: CacheManager for data caching with lineage
- analysis: profiling and diagnostics
- reports: module-level reporting artifacts
- metrics: evaluation and scoring utilities
- utils: helpers and notebook utilities

Quick Start:
    from forecast_foundations import load_m5, CacheManager

    cache = CacheManager(Path("data/cache"))
    df = load_m5(Path("data"), cache=cache, messify=True)
"""

# =============================================================================
# Data Loading
# =============================================================================
from .loaders import (
    load_m5,
    load_m5_with_feedback,
    load_m5_calendar,
    has_m5_cache,
    create_unique_id,
    expand_hierarchy,
    create_subset,
    messify_m5_data,
    HIERARCHY_COLS,
    M5_AVAILABLE,
)

from .features import aggregate_calendar_to_weekly
from .version import __version__

# =============================================================================
# Cache Management
# =============================================================================
from .cache.cache import CacheManager, ArtifactManager

# =============================================================================
# Reporting & Analysis
# =============================================================================
from .analysis.profile import (
    acf_summary,
    profile_series,
    profile_dataframe,
    summarize_profiles,
    interpret_strength,
    calc_stl_strength,
    calc_acf_metrics,
    calc_intermittency_metrics,
    calc_distribution_metrics,
    calc_volatility_metrics,
    calc_outlier_metrics,
    ADI_THRESHOLD,
    CV2_THRESHOLD,
)

# =============================================================================
# Reports
# =============================================================================
from .reports import (
    ModuleReport,
    Snapshot,
    MODULE_CHECKS,
    MODULE_TITLES,
)


# =============================================================================
# Helpers
# =============================================================================
from .utils.helpers import (
    find_project_root,
    get_notebook_name,
    get_notebook_path,
    get_module_from_notebook,
    get_artifact_subfolder,
    plot_ld6_vs_sb,
    ld6_vs_sb_summary
)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Data Loading
    "load_m5",
    "load_m5_with_feedback",
    "load_m5_calendar",
    "has_m5_cache",
    "create_unique_id",
    "expand_hierarchy",
    "create_subset",
    "messify_m5_data",
    "aggregate_calendar_to_weekly",
    "HIERARCHY_COLS",
    "M5_AVAILABLE",
    # Cache
    "CacheManager",
    "ArtifactManager",
    # Profiling / Analysis
    "profile_series",
    "profile_dataframe",
    "summarize_profiles",
    "interpret_strength",
    "calc_stl_strength",
    "calc_acf_metrics",
    "calc_intermittency_metrics",
    "calc_distribution_metrics",
    "calc_volatility_metrics",
    "calc_outlier_metrics",
    "ADI_THRESHOLD",
    "CV2_THRESHOLD",
    # Reports
    "ModuleReport",
    "Snapshot",
    "MODULE_CHECKS",
    "MODULE_TITLES",
    # Metrics
    "MetricsCalculator",
    # Helpers
    "find_project_root",
    "get_notebook_name",
    "get_notebook_path",
    "get_module_from_notebook",
    "get_artifact_subfolder",
    "plot_ld6_vs_sb",
    "ld6_vs_sb_summary",
    # Theme
    "theme",
]

# =============================================================================
# Notebook bootstrap
# =============================================================================
from .utils.bootstrap import setup_notebook, NotebookEnvironment
