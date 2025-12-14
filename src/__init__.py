"""
Forecast Academy
================

Core modules for the Real-World Forecasting Foundations course.

Structure:
- src.data: M5 data loading, preprocessing, messification
- src.cache: CacheManager for data caching with lineage
- src.report: FirstContactReport for data quality assessment
- src.helpers: Utility functions (path finding, notebook detection)

Quick Start:
    from src import load_m5, CacheManager, first_contact_check
    
    cache = CacheManager(Path('data/cache'))
    df = load_m5(Path('data'), cache=cache, messify=True)
    report = first_contact_check(df, dataset_name='M5 Sales')
"""

# =============================================================================
# Data Loading
# =============================================================================
from .adapters import (
    # Loading
    load_m5,
    load_m5_with_feedback,
    load_m5_calendar,
    has_m5_cache,
    # Preprocessing
    create_unique_id,
    expand_hierarchy,
    create_subset,
    messify_m5_data,
    # Inspection
    check_gaps,
    aggregate_calendar_to_weekly,
    # Constants
    HIERARCHY_COLS,
    M5_AVAILABLE,
)

# =============================================================================
# Cache Management
# =============================================================================
from .cache.cache import CacheManager

# =============================================================================
# Reporting
# =============================================================================
from .analysis.report import first_contact_check, FirstContactReport

# =============================================================================
# Helpers
# =============================================================================
from .utils.helpers import (
    find_project_root,
    get_notebook_name,
    get_module_from_notebook,
)


__all__ = [
    # Data Loading
    'load_m5',
    'load_m5_with_feedback',
    'load_m5_calendar',
    'has_m5_cache',
    # Preprocessing
    'create_unique_id',
    'expand_hierarchy',
    'create_subset',
    'messify_m5_data',
    # Inspection
    'check_gaps',
    'aggregate_calendar_to_weekly',
    # Constants
    'HIERARCHY_COLS',
    'M5_AVAILABLE',
    # Cache
    'CacheManager',
    # Report
    'first_contact_check',
    'FirstContactReport',
    # Helpers
    'find_project_root',
    'get_notebook_name',
    'get_module_from_notebook',
]
