"""
CacheManager with automatic source/lineage tracking
====================================================

Tracks what was loaded in the session and auto-links sources.
"""

import json
import hashlib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from .cache_manager import *

MANIFEST_FILENAME = 'cache_manifest.json'

# Module-level load history (shared across all CacheManager instances)
_load_history: List[str] = []

# M5 hierarchy columns in order
HIERARCHY_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

# def get_notebook_name() -> Optional[str]:
#     """Detect current notebook name."""
#     try:
#         from IPython import get_ipython
#         ipython = get_ipython()
#         if ipython and hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
#             return Path(ipython.user_ns['__vsc_ipynb_file__']).stem
#     except Exception:
#         pass
#     return None


# def get_module_from_notebook() -> Optional[str]:
#     """Extract module (first 4 chars) from notebook name."""
#     nb_name = get_notebook_name()
#     return nb_name[:4] if nb_name and len(nb_name) >= 4 else None


# @dataclass
# class CacheEntry:
#     """Metadata for a cached dataset."""
#     key: str
#     filename: str
#     module: str
#     config: Dict[str, Any]
#     config_hash: str
#     created_at: str
#     rows: int
#     columns: List[str]
#     size_mb: float
#     source: Optional[str] = None
#     report_filename: Optional[str] = None


# class CacheManager:
#     """
#     Manages cached datasets with automatic lineage tracking.
    
#     Lineage is tracked automatically:
#     - Every load() call is recorded
#     - save() auto-links to the most recent load as source
    
#     Examples
#     --------
#     >>> cache = CacheManager(DATA_DIR / 'cache')
#     >>> outputs = CacheManager(DATA_DIR / 'outputs')
#     >>> 
#     >>> # Load tracks automatically
#     >>> df = load_m5(DATA_DIR, cache=cache, cache_key='m5_messified', ...)
#     >>> 
#     >>> # Source auto-detected from last load
#     >>> outputs.save(df=weekly_sales, report=report, config={...})
#     >>> # → source='m5_messified' (automatic!)
#     """
    
#     def __init__(self, cache_dir: Path):
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.manifest_path = self.cache_dir / MANIFEST_FILENAME
#         self._manifest = self._load_manifest()
    
#     def _load_manifest(self) -> Dict[str, dict]:
#         if self.manifest_path.exists():
#             with open(self.manifest_path, 'r') as f:
#                 return json.load(f)
#         return {}
    
#     def _save_manifest(self):
#         with open(self.manifest_path, 'w') as f:
#             json.dump(self._manifest, f, indent=2, default=str)
    
#     @staticmethod
#     def _hash_config(config: dict) -> str:
#         config_str = json.dumps(config, sort_keys=True, default=str)
#         return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
#     @staticmethod
#     def last_loaded() -> Optional[str]:
#         """Return the most recently loaded cache key (across all managers)."""
#         return _load_history[-1] if _load_history else None
    
#     @staticmethod
#     def load_history() -> List[str]:
#         """Return full load history for this session."""
#         return _load_history.copy()
    
#     @staticmethod
#     def clear_history():
#         """Clear load history (e.g., at start of notebook)."""
#         _load_history.clear()
    
#     def save(
#         self,
#         df: pd.DataFrame,
#         key: Optional[str] = None,
#         config: Optional[Dict[str, Any]] = None,
#         module: Optional[str] = None,
#         source: Optional[str] = None,  # ← Auto-detected if None
#         report: Optional['FirstContactReport'] = None,
#         overwrite: bool = True
#     ) -> Path:
#         """
#         Save DataFrame with automatic source tracking.
        
#         Parameters
#         ----------
#         df : pd.DataFrame
#             Data to save
#         key : str, optional
#             Cache key. Defaults to '{notebook_name}_output'
#         config : dict, optional
#             Configuration for cache invalidation
#         module : str, optional
#             Module identifier. Auto-detects from notebook.
#         source : str, optional
#             Parent cache key. If None, auto-detects from last load.
#         report : FirstContactReport, optional
#             Report to save alongside data
#         overwrite : bool, default=True
#             Overwrite existing cache
#         """
#         # Auto-detect key
#         if key is None:
#             nb_name = get_notebook_name()
#             if nb_name:
#                 key = f"{nb_name}_output"
#             else:
#                 raise ValueError("Could not auto-detect key. Provide explicitly.")
        
#         # Auto-detect module
#         if module is None:
#             module = get_module_from_notebook() or 'unknown'
        
#         # Auto-detect source from load history
#         if source is None:
#             source = self.last_loaded()
        
#         # Default empty config
#         if config is None:
#             config = {}
        
#         if key in self._manifest and not overwrite:
#             print(f"⚠ Cache '{key}' exists. Use overwrite=True to replace.")
#             return self.cache_dir / self._manifest[key]['filename']
        
#         # Save data
#         config_hash = self._hash_config(config)
#         filename = f"{key}_{config_hash}.parquet"
#         filepath = self.cache_dir / filename
#         df.to_parquet(filepath, index=False)
        
#         # Save report if provided
#         report_filename = None
#         if report is not None:
#             report_filename = f"{key}_report.csv"
#             report_path = self.cache_dir / report_filename
#             report.checks.to_csv(report_path, index=False)
        
#         # Create manifest entry
#         entry = CacheEntry(
#             key=key,
#             filename=filename,
#             module=module,
#             config=config,
#             config_hash=config_hash,
#             created_at=datetime.now().isoformat(),
#             rows=len(df),
#             columns=list(df.columns),
#             size_mb=round(filepath.stat().st_size / 1024**2, 2),
#             source=source,
#             report_filename=report_filename
#         )
        
#         self._manifest[key] = asdict(entry)
#         self._save_manifest()
        
#         # Output
#         print(f"✓ Saved '{key}'")
#         print(f"   Data:   {filename} ({entry.size_mb} MB, {entry.rows:,} rows)")
#         if report_filename:
#             print(f"   Report: {report_filename}")
#         if source:
#             print(f"   Source: {source} (auto-detected)" if source == self.last_loaded() else f"   Source: {source}")
        
#         return filepath
    
#     def load(
#         self,
#         key: str,
#         config: Optional[Dict[str, Any]] = None,
#         with_report: bool = False,
#         verbose: bool = True
#     ):
#         """
#         Load DataFrame from cache (and track for lineage).
        
#         Parameters
#         ----------
#         key : str
#             Cache key to load
#         config : dict, optional
#             Validates against stored config
#         with_report : bool, default=False
#             Return (df, report_df) tuple
#         verbose : bool, default=True
#             Print loading info
#         """
#         if key not in self._manifest:
#             if verbose:
#                 print(f"⚠ Cache '{key}' not found")
#             return (None, None) if with_report else None
        
#         entry = self._manifest[key]
        
#         # Config validation
#         if config is not None:
#             current_hash = self._hash_config(config)
#             if current_hash != entry['config_hash']:
#                 if verbose:
#                     print(f"⚠ Cache '{key}' config mismatch - will regenerate")
#                 return (None, None) if with_report else None
        
#         # Load data
#         filepath = self.cache_dir / entry['filename']
#         if not filepath.exists():
#             if verbose:
#                 print(f"⚠ Cache file missing: {filepath}")
#             return (None, None) if with_report else None
        
#         df = pd.read_parquet(filepath)
        
#         # Track this load for lineage
#         _load_history.append(key)
        
#         if verbose:
#             print(f"✓ Loaded '{key}' from cache")
#             print(f"   Module: {entry['module']} | Created: {entry['created_at'][:10]}")
#             print(f"   Shape: {entry['rows']:,} × {len(entry['columns'])}")
        
#         # Load report if requested
#         if with_report:
#             report_df = None
#             if entry.get('report_filename'):
#                 report_path = self.cache_dir / entry['report_filename']
#                 if report_path.exists():
#                     report_df = pd.read_csv(report_path)
#                     if verbose:
#                         print(f"   Report: {entry['report_filename']}")
#             return df, report_df
        
#         return df
    
#     def exists(self, key: str, config: Optional[Dict[str, Any]] = None) -> bool:
#         if key not in self._manifest:
#             return False
#         if config is not None:
#             return self._hash_config(config) == self._manifest[key]['config_hash']
#         return True
    
#     def info(self, key: str) -> Optional[dict]:
#         """Print detailed info about a cached dataset."""
#         if key not in self._manifest:
#             print(f"⚠ Cache '{key}' not found")
#             return None
        
#         entry = self._manifest[key]
#         print(f"\n{'='*60}")
#         print(f"CACHE: {key}")
#         print(f"{'='*60}")
#         print(f"  Data:     {entry['filename']}")
#         print(f"  Report:   {entry.get('report_filename', 'None')}")
#         print(f"  Module:   {entry['module']}")
#         print(f"  Created:  {entry['created_at']}")
#         print(f"  Size:     {entry['size_mb']} MB")
#         print(f"  Shape:    {entry['rows']:,} × {len(entry['columns'])}")
#         if entry.get('source'):
#             print(f"  Source:   {entry['source']}")
#         print(f"\n  Config:")
#         for k, v in entry['config'].items():
#             print(f"    {k}: {v}")
#         print(f"{'='*60}\n")
#         return entry
    
#     def list(self) -> pd.DataFrame:
#         """List all cached datasets."""
#         if not self._manifest:
#             print("📦 No cached datasets found.")
#             return pd.DataFrame()
        
#         rows = [{
#             'Key': key,
#             'Module': e['module'],
#             'Rows': f"{e['rows']:,}",
#             'Size (MB)': e['size_mb'],
#             'Report': '✓' if e.get('report_filename') else '-',
#             'Source': e.get('source', '-')
#         } for key, e in self._manifest.items()]
        
#         df = pd.DataFrame(rows)
#         print(f"\n📦 Cached Datasets ({len(df)}):\n")
#         print(df.to_string(index=False))
#         return df
    
#     def lineage(self, key: str) -> list:
#         """Show data lineage for a cached dataset."""
#         if key not in self._manifest:
#             print(f"⚠ Cache '{key}' not found")
#             return []
        
#         chain = [key]
#         current = key
#         while True:
#             entry = self._manifest.get(current)
#             if not entry or not entry.get('source'):
#                 break
#             chain.append(entry['source'])
#             current = entry['source']
        
#         print(f"\n📜 Lineage for '{key}':")
#         for i, item in enumerate(reversed(chain)):
#             indent = "  " * i
#             arrow = "→ " if i > 0 else ""
#             module = self._manifest.get(item, {}).get('module', '?')
#             has_report = '📋' if self._manifest.get(item, {}).get('report_filename') else ''
#             print(f"   {indent}{arrow}{item} ({module}) {has_report}")
#         return list(reversed(chain))
    
#     def delete(self, key: str):
#         """Delete a cached dataset and its report."""
#         if key not in self._manifest:
#             print(f"⚠ Cache '{key}' not found")
#             return
        
#         entry = self._manifest[key]
        
#         filepath = self.cache_dir / entry['filename']
#         if filepath.exists():
#             filepath.unlink()
        
#         if entry.get('report_filename'):
#             report_path = self.cache_dir / entry['report_filename']
#             if report_path.exists():
#                 report_path.unlink()
        
#         del self._manifest[key]
#         self._save_manifest()
#         print(f"✓ Deleted cache '{key}'")


# # =============================================================================
# # USAGE
# # =============================================================================

# """
# # Setup
# cache = CacheManager(DATA_DIR / 'cache')
# outputs = CacheManager(DATA_DIR / 'outputs')

# # Load (automatically tracked)
# daily_sales = load_m5(
#     DATA_DIR,
#     cache=cache,
#     cache_key='m5_messified',  # ← This gets tracked
#     messify=True,
#     ...
# )

# # ... process data ...

# # Save (source auto-detected!)
# outputs.save(
#     df=weekly_sales_opt,
#     report=report,
#     config={...}
#     # source='m5_messified'  ← No longer needed! Auto-detected.
# )
# # ✓ Saved '1_06_first_contact_output'
# #    Data:   1_06_first_contact_output_a1b2.parquet
# #    Report: 1_06_first_contact_output_report.csv
# #    Source: m5_messified (auto-detected)

# # Check what was loaded this session
# CacheManager.load_history()
# # ['m5_messified']

# # Override auto-detection if needed
# outputs.save(df=df, source='custom_source', ...)
# """

def load_m5(
    data_dir: Path,
    # Caching
    cache: Optional[CacheManager] = None,
    cache_key: str = 'm5_data',
    module: Optional[str] = None,
    force_refresh: bool = False,
    # Data source
    from_parquet: Optional[Path] = None,
    n_series: Optional[int] = None,
    random_state: int = 42,
    # Messification
    messify: bool = False,
    messify_config: Optional[dict] = None,
    # Output format
    include_hierarchy: bool = False,
    # Other
    m5_data_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load M5 time series data with optional messification and caching.
    
    Parameters
    ----------
    data_dir : Path
        Directory for data operations
    cache : CacheManager, optional
        If provided, enables caching. Pass None to disable.
    cache_key : str, default='m5_data'
        Identifier for this dataset in the cache
    module : str, optional
        Module identifier. Auto-detects from notebook filename if None.
    force_refresh : bool, default=False
        Ignore cache and regenerate
    from_parquet : Path, optional
        Load from parquet instead of raw M5
    n_series : int, optional
        Subset to this many series
    random_state : int, default=42
        Random seed
    messify : bool, default=False
        Apply messification
    messify_config : dict, optional
        Override default messification parameters
    include_hierarchy : bool, default=False
        Expand unique_id to hierarchy columns
    m5_data_dir : Path, optional
        Directory for raw M5 data
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    pd.DataFrame
        M5 data
        
    Examples
    --------
    >>> # Without caching
    >>> df = load_m5(Path('data'), messify=True)
    
    >>> # With caching
    >>> cache = CacheManager(Path('data/cache'))
    >>> df = load_m5(
    ...     Path('data'),
    ...     cache=cache,
    ...     cache_key='m5_messified',
    ...     messify=True,
    ...     include_hierarchy=True
    ... )
    """
    
    # Auto-detect module
    if module is None:
        module = get_module_from_notebook() or 'unknown'
    
    # Build messify config
    _messify_config = {
        'random_state': random_state,
        'zeros_to_na_frac': 0.15,
        'zeros_drop_frac': 0.02,
        'zeros_drop_gaps_frac': None,
        'duplicates_add_n': 150,
        'na_drop_frac': None,
        'dtypes_corrupt': True,
    }
    if messify_config:
        _messify_config.update(messify_config)
    
    # Full config for cache
    full_config = {
        'from_parquet': str(from_parquet) if from_parquet else None,
        'n_series': n_series,
        'random_state': random_state,
        'messify': messify,
        'include_hierarchy': include_hierarchy,
        **_messify_config
    }
    
    # =========================================================================
    # CHECK CACHE
    # =========================================================================
    if cache is not None and not force_refresh:
        df = cache.load(cache_key, config=full_config, verbose=verbose)
        if df is not None:
            return df
        if verbose:
            print(f"🔄 Cache miss for '{cache_key}' - creating fresh...")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    if m5_data_dir is None:
        project_root = find_project_root()
        m5_data_dir = project_root / 'data'
    m5_data_dir = Path(m5_data_dir)
    m5_data_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print("=" * 70)
        print(f"LOADING {'FROM PARQUET' if from_parquet else 'M5 DATA'}")
        print("=" * 70)
    
    S_df = None
    
    if from_parquet:
        from_parquet = Path(from_parquet)
        if not from_parquet.exists():
            raise FileNotFoundError(f"Parquet not found: {from_parquet}")
        df = pd.read_parquet(from_parquet)
        if verbose:
            print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        if include_hierarchy:
            Y_df, _, S_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=True)
        else:
            Y_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=False)
        df = Y_df
    
    has_unique_id = 'unique_id' in df.columns
    has_hierarchy = all(c in df.columns for c in ['item_id', 'store_id'])
    
    # =========================================================================
    # SUBSET
    # =========================================================================
    if n_series is not None:
        if verbose:
            print(f"\n📊 Subsetting to {n_series} series...")
        df = create_subset(df, n_series=n_series, random_state=random_state, verbose=verbose)
    
    # =========================================================================
    # MESSIFY (needs unique_id)
    # =========================================================================
    if messify:
        if verbose:
            print(f"\n🔧 Applying messification...")
        df = messify_m5_data(df, **_messify_config, verbose=verbose)
    
    # =========================================================================
    # EXPAND HIERARCHY (after messify)
    # =========================================================================
    if include_hierarchy and not has_hierarchy:
        if verbose:
            print(f"\n🏗️ Expanding hierarchy...")
        df = expand_hierarchy(df, S_df=S_df, drop_unique_id=False, verbose=verbose)
    
    # =========================================================================
    # SAVE TO CACHE
    # =========================================================================
    if cache is not None:
        cache.save(
            df=df,
            key=cache_key,
            config=full_config,
            module=module,
            source='raw_m5' if not from_parquet else Path(from_parquet).name
        )
    
    if verbose:
        print("\n" + "=" * 70)
        print("LOAD COMPLETE")
        print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
        if cache:
            print(f"  Cached: '{cache_key}'")
        print("=" * 70)
    
    return df

def find_project_root(marker_files=('.git', 'pyproject.toml')):
    """Walk up from this module's location until we find a directory with marker files."""
    # Start from this module's location (utils/load_data.py)
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("Could not find project root")

def get_notebook_name() -> Optional[str]:
    """
    Try to detect the current Jupyter notebook name.

    Returns the notebook name without the .ipynb extension, or None if not in a notebook
    or if detection fails.
    """
    try:
        # Try to get notebook name from IPython/Jupyter
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return None

        # For Jupyter notebooks, try to get the notebook path
        # This works in classic Jupyter and JupyterLab
        if hasattr(ipython, 'kernel') and hasattr(ipython.kernel, 'session'):
            # Try getting from kernel connection file
            import json
            import re
            from pathlib import Path as PathlibPath

            # Alternative: check __vsc_ipynb_file__ for VS Code notebooks
            if '__vsc_ipynb_file__' in dir(ipython.user_ns):
                nb_path = ipython.user_ns['__vsc_ipynb_file__']
                return PathlibPath(nb_path).stem

        # Check for VS Code's notebook variable directly in user namespace
        if hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            nb_path = ipython.user_ns['__vsc_ipynb_file__']
            return Path(nb_path).stem

    except Exception:
        pass

    return None

def create_unique_id(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    separator: str = '_',
    target_col: str = 'unique_id',
    inplace: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a unique_id column by concatenating multiple columns.

    Nixtla libraries require a 'unique_id' column to identify each time series.
    This function creates it by joining specified columns with a separator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str, optional
        Columns to concatenate. If None, auto-detects:
        - Uses ['item_id', 'store_id'] if both exist (M5 default)
        - Falls back to all categorical/object columns if not
    separator : str, default='_'
        Separator between column values
    target_col : str, default='unique_id'
        Name of the output column
    inplace : bool, default=False
        If True, modify DataFrame in place and return it.
        If False, return a copy with the new column.
    verbose : bool, default=True
        Print summary of unique_id creation

    Returns
    -------
    pd.DataFrame
        DataFrame with new unique_id column

    Examples
    --------
    >>> # Auto-detect columns (M5 default: item_id + store_id)
    >>> df = create_unique_id(df)

    >>> # Specify columns explicitly
    >>> df = create_unique_id(df, columns=['item_id', 'store_id'])

    >>> # Use different separator
    >>> df = create_unique_id(df, columns=['dept_id', 'state_id'], separator='-')

    >>> # Modify in place
    >>> create_unique_id(df, inplace=True)
    """
    if not inplace:
        df = df.copy()

    # Auto-detect columns if not specified
    if columns is None:
        # Default M5 pattern: item_id + store_id
        if 'item_id' in df.columns and 'store_id' in df.columns:
            columns = ['item_id', 'store_id']
        else:
            # Fall back to categorical/object columns (excluding common non-ID cols)
            exclude = {'ds', 'y', 'date', 'value', 'target', target_col}
            columns = [
                c for c in df.columns
                if c not in exclude and
                (df[c].dtype == 'object' or df[c].dtype.name == 'category')
            ]
            if not columns:
                raise ValueError(
                    "No columns specified and could not auto-detect. "
                    "Please provide columns=['col1', 'col2', ...]"
                )

    # Validate columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Create unique_id by concatenating columns
    df[target_col] = df[columns[0]].astype(str)
    for col in columns[1:]:
        df[target_col] = df[target_col] + separator + df[col].astype(str)

    if verbose:
        n_unique = df[target_col].nunique()
        print(f"Created {target_col}: {n_unique:,} unique series")
        print(f"Sample: {df[target_col].iloc[0]}")

    return df


def expand_hierarchy(
    df: pd.DataFrame,
    S_df: pd.DataFrame = None,
    id_col: str = 'unique_id',
    drop_unique_id: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Replace unique_id with original M5 hierarchy columns.
    
    The unique_id in M5 (e.g., "FOODS_1_001_CA_1") encodes the hierarchy:
    - item_id: FOODS_1_001
    - dept_id: FOODS_1
    - cat_id: FOODS
    - store_id: CA_1
    - state_id: CA
    
    This function expands unique_id back to these original columns either by
    parsing the ID string or by merging with S_df (static hierarchy dataframe).
    
    OPTIMIZED: Uses vectorized string operations instead of slow apply().
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with unique_id column
    S_df : pd.DataFrame, optional
        Static hierarchy dataframe from load_m5_full(). If provided, uses merge.
        If None, parses the unique_id string directly.
    id_col : str, default='unique_id'
        Name of the ID column
    drop_unique_id : bool, default=True
        Whether to drop the unique_id column after expansion
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchy columns (item_id, dept_id, cat_id, store_id, state_id)
        
    Examples
    --------
    >>> df = load_m5(Path('data'))
    >>> df_expanded = expand_hierarchy(df)
    >>> print(df_expanded.columns)
    # ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'ds', 'y']
    """
    if id_col not in df.columns:
        if verbose:
            print(f"⚠ Column '{id_col}' not found, returning unchanged")
        return df
    
    df_result = df.copy()
    
    if S_df is not None:
        # Use merge with S_df for accurate hierarchy
        if verbose:
            print("  Expanding hierarchy via S_df merge...")
        
        # S_df should have unique_id and hierarchy columns
        hierarchy_cols_present = [c for c in HIERARCHY_COLS if c in S_df.columns]
        merge_cols = [id_col] + hierarchy_cols_present
        
        df_result = df_result.merge(
            S_df[merge_cols].drop_duplicates(),
            on=id_col,
            how='left'
        )
    else:
        # OPTIMIZED: Vectorized parsing using str.split instead of apply()
        # Format: {cat_id}_{dept_num}_{item_num}_{state_id}_{store_num}
        # Example: FOODS_1_001_CA_1 -> cat=FOODS, dept=FOODS_1, item=FOODS_1_001, state=CA, store=CA_1
        if verbose:
            print("  Expanding hierarchy via vectorized ID parsing...")
        
        # Get unique IDs and parse them (do this on unique values for speed)
        unique_ids = df_result[id_col].unique()
        id_mapping = pd.DataFrame({id_col: unique_ids})
        
        # Vectorized split - much faster than apply()
        parts = id_mapping[id_col].str.split('_', expand=True)
        
        # Build hierarchy columns from parts
        # parts[0] = cat (e.g., FOODS)
        # parts[1] = dept_num (e.g., 1)
        # parts[2] = item_num (e.g., 001)
        # parts[3] = state (e.g., CA)
        # parts[4] = store_num (e.g., 1)
        id_mapping['cat_id'] = parts[0]
        id_mapping['dept_id'] = parts[0] + '_' + parts[1]
        id_mapping['item_id'] = parts[0] + '_' + parts[1] + '_' + parts[2]
        id_mapping['state_id'] = parts[3]
        id_mapping['store_id'] = parts[3] + '_' + parts[4]
        
        # Merge back
        df_result = df_result.merge(id_mapping, on=id_col, how='left')
    
    # Reorder columns: hierarchy first, then rest
    other_cols = [c for c in df_result.columns if c not in HIERARCHY_COLS + [id_col]]
    if drop_unique_id:
        new_order = HIERARCHY_COLS + other_cols
    else:
        new_order = HIERARCHY_COLS + [id_col] + other_cols
    
    # Only include columns that exist
    new_order = [c for c in new_order if c in df_result.columns]
    df_result = df_result[new_order]
    
    if verbose:
        print(f"  ✓ Added hierarchy columns: {HIERARCHY_COLS}")
    
    return df_result


# ============================================================================
# SECTION 1: LOADING FUNCTIONS
# ============================================================================

def has_m5_cache(data_dir: Path) -> bool:
    """
    Quick check for M5 cache files.
    
    Checks for specific M5 cache files rather than doing expensive
    recursive directory searches.
    
    Parameters
    ----------
    data_dir : Path
        Directory where M5 data would be cached
        
    Returns
    -------
    bool
        True if M5 cache files are found, False otherwise
        
    Examples
    --------
    >>> data_dir = Path('data')
    >>> if has_m5_cache(data_dir):
    ...     print("Using cached M5")
    ... else:
    ...     print("Will download M5")
    """
    # M5.load() caches to these file patterns
    cache_files = [
        'M5.parquet',
        'M5.csv',
        'm5.parquet',
        'm5.csv',
        'm5.p',
        'M5.p'
    ]

    # Also check common subdirectories and nested paths
    cache_subdirs = ['cache', 'm5', 'M5', 'm5-forecasting-accuracy', 'm5/datasets', 'M5/datasets']

    # Check main directory
    if any((data_dir / f).exists() for f in cache_files):
        return True

    # Check subdirectories
    for subdir in cache_subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            if any((subdir_path / f).exists() for f in cache_files):
                return True

    return False


def load_m5_calendar(
    data_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load M5 calendar data with date information and events.

    The calendar contains:
    - date: Actual date
    - wm_yr_wk: Walmart year-week (for price joining)
    - weekday: Day name
    - wday: Day number (1-7)
    - month, year: Date components
    - d: Day code (d_1, d_2, ..., d_1969)
    - event_name_1/2: Holiday/event names
    - event_type_1/2: Event types (Cultural, National, Religious, Sporting)
    - snap_CA/TX/WI: SNAP (food stamps) indicators by state

    Parameters
    ----------
    data_dir : Path
        Directory containing M5 data (will look for calendar.csv)
    verbose : bool, default=True
        Whether to print loading information

    Returns
    -------
    pd.DataFrame
        Calendar DataFrame with date and event information

    Raises
    ------
    FileNotFoundError
        If calendar.csv is not found in expected locations

    Examples
    --------
    >>> calendar = load_m5_calendar(Path('data'))
    >>> calendar['date'] = pd.to_datetime(calendar['date'])
    >>> print(calendar.columns)
    # ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd',
    #  'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    #  'snap_CA', 'snap_TX', 'snap_WI']
    """
    data_dir = Path(data_dir)

    # Search for calendar.csv in common M5 cache locations
    search_paths = [
        data_dir / 'calendar.csv',
        data_dir / 'm5' / 'calendar.csv',
        data_dir / 'M5' / 'calendar.csv',
        data_dir / 'm5-forecasting-accuracy' / 'calendar.csv',
        data_dir / 'datasets' / 'calendar.csv',
        data_dir / 'm5' / 'datasets' / 'calendar.csv',
        data_dir / 'M5' / 'datasets' / 'calendar.csv',
    ]

    calendar_path = None
    for path in search_paths:
        if path.exists():
            calendar_path = path
            break

    if calendar_path is None:
        raise FileNotFoundError(
            f"calendar.csv not found in data directory.\n"
            f"Searched locations:\n" +
            "\n".join(f"  - {p}" for p in search_paths[:4]) +
            f"\n\nMake sure M5 data has been downloaded. "
            f"You can trigger the download by calling load_m5(data_dir) first."
        )

    if verbose:
        print(f"Loading calendar from: {calendar_path}")

    calendar = pd.read_csv(calendar_path)

    if verbose:
        print(f"  Shape: {calendar.shape[0]:,} rows × {calendar.shape[1]} columns")
        print(f"  Date range: {calendar['date'].iloc[0]} to {calendar['date'].iloc[-1]}")
        n_events = calendar['event_name_1'].notna().sum()
        print(f"  Events: {n_events} days with events")

    return calendar


def load_m5_with_feedback(
    data_dir: Path,
    verbose: bool = True,
    return_additional: bool = False
) -> Tuple:
    """
    Load M5 dataset with informative feedback.
    
    M5 includes 3 separate dataframes:
    - Y_df: Time series data (unique_id, ds, y) - what forecasters need
    - X_df: Calendar/exogenous features (events, SNAP, holidays)
    - S_df: Static hierarchy (item → dept → category, store → state)
    
    By default, returns only Y_df since that's needed 99% of the time.
    Set return_additional=True to get all 3.
    
    Provides timing, memory usage, and cache status information.
    Optimized for fast cache detection (no recursive directory searches).
    
    Parameters
    ----------
    data_dir : Path
        Directory for M5 data cache
    verbose : bool, default=True
        Whether to print progress messages
    return_additional : bool, default=False
        If True, returns (Y_df, X_df, S_df). If False, returns just Y_df.
        
    Returns
    -------
    pd.DataFrame or tuple
        If return_additional=False: Returns Y_df only
        If return_additional=True: Returns (Y_df, X_df, S_df) tuple
        
    Raises
    ------
    ImportError
        If datasetsforecast is not installed
        
    Examples
    --------
    >>> # Just get time series data (default)
    >>> df = load_m5_with_feedback(Path('data'))
    >>> print(df.columns)  # ['unique_id', 'ds', 'y']
    
    >>> # Get all 3 dataframes
    >>> Y_df, X_df, S_df = load_m5_with_feedback(Path('data'), return_additional=True)
    >>> # Y_df: time series | X_df: calendar features | S_df: hierarchy
    
    >>> # Silent loading (no output)
    >>> df = load_m5_with_feedback(Path('data'), verbose=False)
    """
    if not M5_AVAILABLE:
        raise ImportError(
            "datasetsforecast is required to load M5 data.\n"
            "Install it with: pip install datasetsforecast"
        )
    
    # Fast cache check (4-8 file checks, no recursion)
    cache_exists = has_m5_cache(data_dir)
    
    if verbose:
        if cache_exists:
            print("✓ M5 cache detected. Loading from local files...")
        else:
            print("⚠ No M5 cache found. First download will take ~30-60s (~200MB)...")
            print("  Subsequent loads will be instant.")
    
    # Load with timing
    start_time = time.time()
    result = M5.load(directory=str(data_dir))  # Returns (Y_df, X_df, S_df)
    load_time = time.time() - start_time
    
    # Extract Y_df (time series data) for reporting
    df = result[0]
    
    if verbose:
        print(f"✓ Loaded in {load_time:.1f}s")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory: {memory_mb:,.1f} MB")
        
        # Show columns
        n_cols_show = min(8, len(df.columns))
        cols_str = ', '.join(df.columns[:n_cols_show].tolist())
        if len(df.columns) > n_cols_show:
            cols_str += '...'
        print(f"  Columns: {cols_str}")
        
        # Clarify what's being returned
        if return_additional:
            print(f"  Returning: Y_df, X_df, S_df (all 3 dataframes)")
        else:
            print(f"  Returning: Y_df only (time series data)")
    
    # Return based on user preference
    if return_additional:
        return result  # Returns (Y_df, X_df, S_df)
    else:
        return df  # Just return Y_df


"""
Optimized load_m5() - Drop-in replacement for the loading section

Key optimizations:
1. Check for unique_id BEFORE deciding to create it
2. Check for hierarchy columns BEFORE deciding to expand
3. Clear verbose output showing what's skipped vs applied
"""

# def load_m5(
#     data_dir: Path,
#     from_parquet: Optional[Path] = None,
#     verbose: bool = True,
#     messify: bool = False,
#     messify_kwargs: Optional[dict] = None,
#     include_hierarchy: bool = False,
#     create_unique_id: bool = True,
#     n_series: Optional[int] = None,
#     random_state: int = 42,
#     m5_data_dir: Optional[Path] = None
# ) -> pd.DataFrame:
#     """
#     Load M5 time series data with optional preprocessing.
    
#     OPTIMIZED: Skips redundant operations by checking existing columns first.
#     """
    
#     # Determine source
#     loading_from_parquet = from_parquet is not None

#     # Default m5_data_dir to project root's data folder
#     if m5_data_dir is None:
#         project_root = find_project_root()
#         m5_data_dir = project_root / 'data'
#     m5_data_dir = Path(m5_data_dir)
#     m5_data_dir.mkdir(exist_ok=True, parents=True)

#     if verbose:
#         print("=" * 70)
#         if loading_from_parquet:
#             print(f"LOADING FROM PARQUET: {Path(from_parquet).name}")
#         else:
#             print("LOADING M5 DATA")
#         print("=" * 70)

#     # =========================================================================
#     # STEP 1: Load data
#     # =========================================================================
#     S_df = None

#     if loading_from_parquet:
#         from_parquet = Path(from_parquet)
#         if not from_parquet.exists():
#             raise FileNotFoundError(f"Parquet file not found: {from_parquet}")

#         start_time = time.time()
#         df = pd.read_parquet(from_parquet)
#         load_time = time.time() - start_time

#         if verbose:
#             print(f"✓ Loaded in {load_time:.1f}s")
#             print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
#             print(f"  Columns: {list(df.columns)}")
#     else:
#         # Load raw M5 - only fetch S_df if we'll need it
#         need_S_df = include_hierarchy and not loading_from_parquet
#         if need_S_df:
#             Y_df, _, S_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=True)
#         else:
#             Y_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=False)
#         df = Y_df

#     # =========================================================================
#     # STEP 2: Detect what already exists (do this ONCE, up front)
#     # =========================================================================
#     has_unique_id = 'unique_id' in df.columns
#     has_hierarchy = all(c in df.columns for c in ['item_id', 'store_id'])
#     has_full_hierarchy = all(c in df.columns for c in HIERARCHY_COLS)
    
#     if verbose:
#         print(f"\n📋 Existing columns check:")
#         print(f"   unique_id:  {'✓ exists' if has_unique_id else '✗ missing'}")
#         print(f"   hierarchy:  {'✓ exists' if has_hierarchy else '✗ missing'}")

#     # =========================================================================
#     # STEP 3: Subset if requested (before messification for speed)
#     # =========================================================================
#     if n_series is not None:
#         if verbose:
#             print(f"\n📊 Subsetting to {n_series} series...")
#         df = create_subset(df, n_series=n_series, random_state=random_state, verbose=verbose)

#     # =========================================================================
#     # STEP 4: Messify if requested
#     # =========================================================================
#     if messify:
#         if verbose:
#             print(f"\n🔧 Applying messification...")
        
#         _messify_kwargs = {
#             'random_state': random_state,
#             'zeros_to_na_frac': 0.15,
#             'zeros_drop_frac': 0.02,
#             'zeros_drop_gaps_frac': None,
#             'duplicates_add_n': 150,
#             'na_drop_frac': None,
#             'dtypes_corrupt': True,
#             'cache_dir': data_dir,
#             'verbose': verbose
#         }
#         if messify_kwargs:
#             _messify_kwargs.update(messify_kwargs)
        
#         df = messify_m5_data(df, **_messify_kwargs)

#     # =========================================================================
#     # STEP 5: Expand hierarchy (SKIP if already exists)
#     # =========================================================================
#     if include_hierarchy:
#         if has_hierarchy:
#             if verbose:
#                 print(f"\n🏗️ Hierarchy expansion: SKIPPED (columns already exist)")
#         else:
#             if verbose:
#                 print(f"\n🏗️ Expanding hierarchy...")
#             df = expand_hierarchy(df, S_df=S_df, verbose=verbose)
#             has_hierarchy = True  # Update flag

#     # =========================================================================
#     # STEP 6: Create unique_id (SKIP if already exists)
#     # =========================================================================
#     if create_unique_id:
#         if has_unique_id:
#             if verbose:
#                 print(f"\n🔑 unique_id creation: SKIPPED (column already exists)")
#         elif not has_hierarchy:
#             if verbose:
#                 print(f"\n🔑 unique_id creation: SKIPPED (no hierarchy columns to build from)")
#         else:
#             if verbose:
#                 print(f"\n🔑 Creating unique_id...")
#             df = globals()['create_unique_id'](
#                 df,
#                 columns=['item_id', 'store_id'],
#                 verbose=verbose
#             )

#     # =========================================================================
#     # SUMMARY
#     # =========================================================================
#     if verbose:
#         print("\n" + "=" * 70)
#         print("LOAD COMPLETE")
#         print("=" * 70)
#         print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
#         print(f"  Columns: {list(df.columns)}")
#         print("=" * 70)

#     return df

# ============================================================================
# SECTION 2: SUBSET CREATION
# ============================================================================

def create_subset(
    df: pd.DataFrame,
    n_series: int = 100,
    id_col: str = 'unique_id',
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a random subset of series for faster processing.
    
    Useful for development, testing, and training when you don't need
    all 30,490 M5 series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full M5 dataset
    n_series : int, default=100
        Number of series to sample
    id_col : str, default='unique_id'
        Name of the series ID column
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print subset information
        
    Returns
    -------
    pd.DataFrame
        Subset of the original data with n_series
        
    Examples
    --------
    >>> # Load full M5
    >>> df_full = load_m5(Path('data'))
    >>> 
    >>> # Create subset with 100 series (much faster!)
    >>> df_subset = create_subset(df_full, n_series=100)
    >>> 
    >>> # Create larger subset
    >>> df_subset = create_subset(df_full, n_series=1000)
    """
    np.random.seed(random_state)
    
    # Get all unique series
    all_series = df[id_col].unique()
    total_series = len(all_series)
    
    # Validate n_series
    if n_series > total_series:
        if verbose:
            print(f"⚠ Requested {n_series} series but only {total_series} available")
            print(f"  Using all {total_series} series")
        n_series = total_series
    
    # Sample series
    sample_series = np.random.choice(all_series, size=n_series, replace=False)
    
    # Create subset
    df_subset = df[df[id_col].isin(sample_series)].copy()
    
    if verbose:
        print("=" * 70)
        print("SUBSET CREATED")
        print("=" * 70)
        print(f"\n📊 SIZE")
        print(f"  Original:  {len(df):>12,} rows, {total_series:>6,} series")
        print(f"  Subset:    {len(df_subset):>12,} rows, {n_series:>6,} series")
        print(f"  Reduction: {len(df_subset) / len(df) * 100:>12.1f}% of original")
        
        # Average observations per series
        if len(df_subset) > 0:
            avg_obs = len(df_subset) / n_series
            print(f"\n📈 STATS")
            print(f"  Avg obs/series: {avg_obs:>8.1f}")
        
        print("\n💡 TIP: Subset is perfect for:")
        print("  • Fast iteration during development")
        print("  • Quick testing of code changes")
        print("  • Training examples in notebooks")
        print("\n  To use full dataset, set: n_series=len(df['unique_id'].unique())")
        print("=" * 70)
    
    return df_subset


# ============================================================================
# SECTION 3: DATA MESSIFICATION
# ============================================================================

def messify_m5_data(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    date_col: str = 'ds',
    target_col: str = 'y',
    random_state: int = 42,
    # --- ZEROS HANDLING ---
    zeros_to_na_frac: Optional[float] = 0.15,
    zeros_drop_frac: Optional[float] = 0.02,
    zeros_drop_gaps_frac: Optional[float] = None,
    # --- DUPLICATES ---
    duplicates_add_n: Optional[int] = 150,
    # --- NA HANDLING ---
    na_drop_frac: Optional[float] = None,
    # --- DATA TYPES ---
    dtypes_corrupt: bool = True,
    # --- CACHING ---
    cache_dir: Optional[Path] = None,
    cache_tag: Optional[str] = None,
    force_refresh: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Messify clean M5 data to simulate real-world data quality issues.
    
    Introduces common data problems found in real forecasting pipelines:
    
    1. zeros_to_na_frac: Convert zeros → NAs (simulates missing reporting)
    2. zeros_drop_frac: Remove zero rows entirely (simulates sparse reporting)
    3. zeros_drop_gaps_frac: Remove zeros from MIDDLE of series (creates internal gaps)
    4. duplicates_add_n: Add duplicate rows (simulates faulty ETL/merges)
    5. na_drop_frac: Drop some NA rows (simulates partial data recovery)
    6. dtypes_corrupt: Convert dates/numbers to strings (simulates CSV round-trips)
    
    Results are cached to speed up repeated runs with the same parameters.
    
    OPTIMIZED: Uses vectorized operations instead of slow groupby-apply patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean M5 dataset from datasetsforecast
    id_col : str, default='unique_id'
        Name of the ID column
    date_col : str, default='ds'
        Name of the date column
    target_col : str, default='y'
        Name of the target column
    random_state : int, default=42
        Random seed for reproducibility
        
    zeros_to_na_frac : float, optional, default=0.15
        Fraction of zero values to convert to NA (0.0 to 1.0).
        Simulates missing data where zeros weren't reported.
        Set to None to disable.
    zeros_drop_frac : float, optional, default=0.02
        Fraction of zero-value rows to drop entirely (0.0 to 1.0).
        Simulates sparse reporting where zero-demand periods aren't recorded.
        Set to None to disable.
    zeros_drop_gaps_frac : float, optional, default=None
        Fraction of zero-value rows to drop from MIDDLE of each series only
        (never first or last row). Creates true internal gaps for testing
        gap detection. Set to None to disable.
    duplicates_add_n : int, optional, default=150
        Number of duplicate rows to add. Simulates faulty ETL or merge issues.
        Set to None to disable.
    na_drop_frac : float, optional, default=None
        Fraction of NA rows to drop (0.0 to 1.0). Applied after zeros_to_na
        creates NAs. Simulates partial data recovery efforts.
        Set to None to disable.
    dtypes_corrupt : bool, default=True
        Whether to corrupt data types by converting date and target columns
        to strings. Simulates CSV round-trips or poorly typed databases.
        
    cache_dir : Path, optional
        Directory to cache the messified data. If None, no caching.
    force_refresh : bool, default=False
        If True, regenerate messified data even if cache exists.
    verbose : bool, default=True
        Whether to print summary of changes.
        
    Returns
    -------
    pd.DataFrame
        Messified version of the input data
        
    Examples
    --------
    >>> # Default messification
    >>> df_messy = messify_m5_data(df_clean)
    
    >>> # Heavy messification with internal gaps
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=0.30,
    ...     zeros_drop_gaps_frac=0.10,
    ...     duplicates_add_n=200,
    ...     cache_dir=Path('data')
    ... )
    
    >>> # Light messification (dtype corruption only)
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=None,
    ...     zeros_drop_frac=None,
    ...     duplicates_add_n=None,
    ...     dtypes_corrupt=True
    ... )
    """
    # Generate cache filename based on parameters
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        # Save cached data to a 'cache' subfolder within the data directory
        cache_subdir = cache_dir / 'cache'
        cache_subdir.mkdir(exist_ok=True, parents=True)

        # Create filename that reflects the messification parameters
        n_series = df[id_col].nunique()
        tag_prefix = f"{cache_tag}_" if cache_tag else ""
        cache_filename = (
            f"{tag_prefix}m5_messy_"
            f"n{n_series}_"
            f"rs{random_state}_"
            f"z2na{int(zeros_to_na_frac*100) if zeros_to_na_frac else 0}_"
            f"zdrp{int(zeros_drop_frac*100) if zeros_drop_frac else 0}_"
            f"zgap{int(zeros_drop_gaps_frac*100) if zeros_drop_gaps_frac else 0}_"
            f"dup{duplicates_add_n if duplicates_add_n else 0}_"
            f"nadrp{int(na_drop_frac*100) if na_drop_frac else 0}_"
            f"dtype{1 if dtypes_corrupt else 0}"
            f".parquet"
        )
        cache_path = cache_subdir / cache_filename
        
        # Check if cache exists and should be used
        if cache_path.exists() and not force_refresh:
            if verbose:
                print("=" * 70)
                print("LOADING CACHED MESSIFIED DATA")
                print("=" * 70)
                print(f"\n📁 Cache file: {cache_path.name}")
                print("   Using cached version (skip messification)")
                print("\n💡 To regenerate: set force_refresh=True")
            
            df_messy = pd.read_parquet(cache_path)
            
            if verbose:
                print(f"\n✓ Loaded {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
                print("=" * 70)
            
            return df_messy
        
        if verbose and cache_path.exists():
            print(f"\n⚠️  Cache exists but force_refresh=True, regenerating...\n")
    
    # Perform messification
    np.random.seed(random_state)
    df_messy = df.copy()
    changes_log = []
    
    # Step 1: Convert some zeros to NAs
    if zeros_to_na_frac is not None and zeros_to_na_frac > 0 and target_col in df_messy.columns:
        if verbose:
            print("Step 1/6: Converting zeros to NAs...")
        
        zero_mask = df_messy[target_col] == 0
        n_zeros = zero_mask.sum()
        n_to_convert = int(n_zeros * zeros_to_na_frac)
        
        if n_to_convert > 0:
            zero_indices = df_messy[zero_mask].index
            na_indices = np.random.choice(zero_indices, size=n_to_convert, replace=False)
            df_messy.loc[na_indices, target_col] = np.nan
            
            changes_log.append(f"Converted {n_to_convert:,} zeros to NAs ({zeros_to_na_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Converted {n_to_convert:,} zeros to NAs")
    elif verbose:
        print("Step 1/6: Converting zeros to NAs... [SKIPPED]")
    
    # Step 2: Add duplicate rows
    if duplicates_add_n is not None and duplicates_add_n > 0:
        if verbose:
            print("Step 2/6: Adding duplicate rows...")
        
        n_duplicates = min(duplicates_add_n, len(df_messy))
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(df_messy.index, size=n_duplicates, replace=False)
            duplicates = df_messy.loc[duplicate_indices].copy()
            df_messy = pd.concat([df_messy, duplicates], ignore_index=True)
            
            changes_log.append(f"Added {n_duplicates:,} duplicate rows")
            if verbose:
                print(f"  ✓ Added {n_duplicates:,} duplicate rows")
    elif verbose:
        print("Step 2/6: Adding duplicate rows... [SKIPPED]")
    
    # Step 3: Remove some zero-demand rows (sparse reporting)
    if zeros_drop_frac is not None and zeros_drop_frac > 0:
        if verbose:
            print("Step 3/6: Dropping zero-demand rows (sparse reporting)...")
        
        # Only target actual zeros (not NAs created in step 1)
        zero_mask = df_messy[target_col] == 0
        zero_indices = df_messy[zero_mask].index
        
        n_to_remove = int(len(zero_indices) * zeros_drop_frac)
        if n_to_remove > 0:
            removal_indices = np.random.choice(zero_indices, size=n_to_remove, replace=False)
            df_messy = df_messy.drop(removal_indices).reset_index(drop=True)
            
            changes_log.append(f"Dropped {n_to_remove:,} zero-demand rows ({zeros_drop_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Dropped {n_to_remove:,} zero-demand rows")
    elif verbose:
        print("Step 3/6: Dropping zero-demand rows... [SKIPPED]")
    
    # Step 4: Drop fraction of NA rows if requested
    if na_drop_frac is not None and na_drop_frac > 0:
        if verbose:
            print("Step 4/6: Dropping NA rows...")

        na_mask = df_messy[target_col].isna()
        n_na = na_mask.sum()

        if n_na > 0:
            n_to_drop = int(n_na * na_drop_frac)
            na_indices = df_messy[na_mask].index
            drop_indices = np.random.choice(na_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)

            changes_log.append(f"Dropped {n_to_drop:,} NA rows ({na_drop_frac*100:.0f}% of NAs)")
            if verbose:
                print(f"  ✓ Dropped {n_to_drop:,} of {n_na:,} NA rows")
        elif verbose:
            print("  No NA rows found to drop")
    elif verbose:
        print("Step 4/6: Dropping NA rows... [SKIPPED]")

    # Step 5: Create internal gaps by dropping ZERO rows from middle of each series
    # OPTIMIZED: Uses vectorized operations instead of slow groupby-apply
    if zeros_drop_gaps_frac is not None and zeros_drop_gaps_frac > 0:
        if verbose:
            print("Step 5/6: Creating internal gaps (dropping middle zeros)...")

        n_before = len(df_messy)
        
        # Sort by id and date (required for identifying first/last per group)
        df_messy = df_messy.sort_values([id_col, date_col]).reset_index(drop=True)
        
        # Vectorized identification of first/last rows per group using shift
        is_first = df_messy[id_col] != df_messy[id_col].shift(1)
        is_last = df_messy[id_col] != df_messy[id_col].shift(-1)
        
        # Middle rows are neither first nor last AND have zero demand
        is_zero = df_messy[target_col] == 0
        is_middle_zero = ~is_first & ~is_last & is_zero
        middle_zero_indices = df_messy.index[is_middle_zero].values
        
        # Sample from middle zero indices
        n_to_drop = int(len(middle_zero_indices) * zeros_drop_gaps_frac)
        
        if n_to_drop > 0 and len(middle_zero_indices) > 0:
            drop_indices = np.random.choice(middle_zero_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)
        
        n_dropped = n_before - len(df_messy)

        changes_log.append(f"Created internal gaps: dropped {n_dropped:,} middle zeros ({zeros_drop_gaps_frac*100:.0f}%)")
        if verbose:
            print(f"  ✓ Dropped {n_dropped:,} zeros from middle of series")
    elif verbose:
        print("Step 5/6: Creating internal gaps... [SKIPPED]")

    # Step 6: Corrupt data types
    if dtypes_corrupt:
        if verbose:
            print("Step 6/6: Corrupting data types...")

        if date_col in df_messy.columns:
            df_messy[date_col] = df_messy[date_col].astype(str)
            changes_log.append(f"Converted {date_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {date_col} to string")

        if target_col in df_messy.columns:
            df_messy[target_col] = df_messy[target_col].astype(str)
            changes_log.append(f"Converted {target_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {target_col} to string")
    elif verbose:
        print("Step 6/6: Corrupting data types... [SKIPPED]")

    # Save to cache if requested
    if cache_dir is not None:
        if verbose:
            print(f"\n💾 Caching messified data...")
            print(f"   → {cache_path.name}")
        
        df_messy.to_parquet(cache_path, index=False)
        
        if verbose:
            cache_size_mb = cache_path.stat().st_size / 1024**2
            print(f"   ✓ Cached ({cache_size_mb:.1f} MB)")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("DATA MESSIFICATION SUMMARY")
        print("=" * 70)
        print(f"\nOriginal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Messified shape: {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
        
        if changes_log:
            print(f"\nChanges applied ({len(changes_log)}):")
            for i, change in enumerate(changes_log, 1):
                print(f"  {i}. {change}")
        else:
            print("\nNo changes applied (all steps skipped)")
        
        print("\n" + "=" * 70)
        print("✓ Data successfully messified!")
        print("=" * 70)
    
    return df_messy


def check_gaps(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    date_col: str = 'ds',
    freq: str = 'W'
) -> bool:
    """
    Diagnose gaps in time series data before filling.

    Compares actual rows per series vs expected rows based on the global
    date range and frequency. Useful to determine if fill_gaps is needed.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with ID and date columns
    id_col : str, default='unique_id'
        Name of the series identifier column
    date_col : str, default='ds'
        Name of the date column
    freq : str, default='W'
        Expected frequency (e.g., 'W', 'W-SUN', 'D', 'M')

    Returns
    -------
    bool
        True if gaps exist and filling is needed, False otherwise

    Examples
    --------
    >>> # Check if gaps exist
    >>> needs_filling = check_gaps(df, freq='W-SUN')
    >>> if needs_filling:
    ...     df = fill_gaps(df, freq='W-SUN')

    >>> # With custom columns
    >>> needs_filling = check_gaps(df, id_col='series_id', date_col='date', freq='D')
    """
    # Get global date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    expected_periods = pd.date_range(min_date, max_date, freq=freq)
    n_expected = len(expected_periods)

    # Count actual rows per series
    actual_counts = df.groupby(id_col).size()
    n_series = len(actual_counts)

    # Series with gaps (fewer rows than expected)
    series_with_gaps = actual_counts[actual_counts < n_expected]
    n_series_with_gaps = len(series_with_gaps)
    total_missing = (n_expected - actual_counts).clip(lower=0).sum()

    print(f"Gap Analysis:")
    print(f"  Date range: {min_date.date()} to {max_date.date()}")
    print(f"  Expected periods per series: {n_expected}")
    print(f"  Series with gaps: {n_series_with_gaps:,} / {n_series:,}")
    print(f"  Total missing rows: {total_missing:,}")

    if n_series_with_gaps > 0:
        print(f"\nWorst offenders (most gaps):")
        gaps = n_expected - series_with_gaps
        for uid, gap_count in gaps.sort_values(ascending=False).head(5).items():
            print(f"  {uid}: {gap_count} missing periods")

    needs_filling = n_series_with_gaps > 0
    print(f"\nNeeds gap filling: {needs_filling}")

    return needs_filling


# ============================================================================
# SECTION 5: INSPECTION UTILITIES
# ============================================================================


# ============================================================================
# SECTION 6: CALENDAR AGGREGATION
# ============================================================================

def aggregate_calendar_to_weekly(
    calendar: pd.DataFrame,
    date_col: str = 'date',
    week_start_day: str = 'Sunday'
) -> pd.DataFrame:
    """
    Aggregate daily calendar data to weekly, splitting events into separate columns.

    Converts daily M5 calendar features (events, SNAP flags) into weekly features
    that align with weekly sales data. Events are split into individual columns
    (event_name_1, event_name_2, etc.) based on the maximum events in any single week.

    Aggregation Rules:
    - Calendar identifiers (wm_yr_wk, month, year): First day of week (Sunday)
    - Events: Collect unique events, split into separate columns
    - SNAP flags: Max (if ANY day has SNAP=1, the week gets 1)

    Parameters
    ----------
    calendar : pd.DataFrame
        Daily calendar data from load_m5_calendar() with columns:
        date, wm_yr_wk, month, year, event_name_1, event_type_1,
        event_name_2, event_type_2, snap_CA, snap_TX, snap_WI
    date_col : str, default='date'
        Name of the date column
    week_start_day : str, default='Sunday'
        Day that starts the week ('Sunday' for Walmart fiscal week)

    Returns
    -------
    pd.DataFrame
        Weekly calendar with columns:
        - ds: Week start date (datetime)
        - wm_yr_wk, month, year: Calendar identifiers
        - event_name_1, event_name_2, ...: Individual event names (one per column)
        - event_type_1, event_type_2, ...: Individual event types (one per column)
        - snap_CA, snap_TX, snap_WI: SNAP flags (1 if any day in week had SNAP)

    Examples
    --------
    >>> calendar = load_m5_calendar(Path('data'))
    >>> calendar['date'] = pd.to_datetime(calendar['date'])
    >>> weekly_cal = aggregate_calendar_to_weekly(calendar)
    >>> print(weekly_cal.columns)
    # ['ds', 'wm_yr_wk', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI',
    #  'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
    """
    df = calendar.copy()

    # Ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Calculate week start (Sunday) for each date
    # dayofweek: Monday=0, ..., Saturday=5, Sunday=6
    df['week_start'] = df[date_col] - pd.to_timedelta(
        (df[date_col].dt.dayofweek + 1) % 7, unit='D'
    )

    # Step 1: Aggregate to weekly with combined event strings
    weekly = df.groupby('week_start').agg({
        'wm_yr_wk': 'first',
        'month': 'first',
        'year': 'first',
        'event_name_1': lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else None,
        'event_type_1': lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else None,
        'event_name_2': lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else None,
        'event_type_2': lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else None,
        'snap_CA': 'max',
        'snap_TX': 'max',
        'snap_WI': 'max',
    }).reset_index()

    # Step 2: Merge event_name_1 and event_name_2 into single list
    weekly['_all_events'] = (
        weekly['event_name_1'].fillna('') + ', ' + weekly['event_name_2'].fillna('')
    ).str.strip(', ').replace('', None)

    weekly['_all_types'] = (
        weekly['event_type_1'].fillna('') + ', ' + weekly['event_type_2'].fillna('')
    ).str.strip(', ').replace('', None)

    # Step 3: Split into separate columns
    # Count max events in any week to determine number of columns needed
    max_events = weekly['_all_events'].str.count(',').max() + 1

    event_name_cols = weekly['_all_events'].str.split(', ', expand=True)
    event_name_cols.columns = [f'event_name_{i+1}' for i in range(event_name_cols.shape[1])]

    event_type_cols = weekly['_all_types'].str.split(', ', expand=True)
    event_type_cols.columns = [f'event_type_{i+1}' for i in range(event_type_cols.shape[1])]

    # Step 4: Build final dataframe
    weekly = weekly.drop(columns=[
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2',
        '_all_events', '_all_types'
    ])
    weekly = pd.concat([weekly, event_name_cols, event_type_cols], axis=1)

    # Rename week_start to ds (standard Nixtla column name)
    weekly = weekly.rename(columns={'week_start': 'ds'})

    return weekly


# ============================================================================
# MODULE EXPORTS
# ============================================================================

# __all__ = [
#     'load_m5',
#     'load_m5_with_feedback',
#     'has_m5_cache',
#     'create_subset',
#     'create_unique_id',
#     'messify_m5_data',
#     'expand_hierarchy',
#     'check_gaps',
#     'find_project_root',
#     'get_notebook_name',
#     'HIERARCHY_COLS',
# ]


# ============================================================================
# MODULE TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    """
    Test/demo the utilities.
    """
    print("M5 Utilities Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("\n  Loading:")
    print("    - load_m5(data_dir, verbose=True, messify=False,")
    print("              include_hierarchy=False, n_series=None)")
    print("        → Main entry point with preprocessing options")
    print("    - load_m5_with_feedback(data_dir, verbose=True, return_additional=False)")
    print("        → Low-level function with full control")
    print("    - has_m5_cache(data_dir)")
    print("        → Check if M5 is already cached")
    print("\n  Preprocessing:")
    print("    - expand_hierarchy(df, S_df=None)")
    print("        → Replace unique_id with item_id, dept_id, cat_id, store_id, state_id")
    print("\n  Subsetting:")
    print("    - create_subset(df, n_series=100, random_state=42)")
    print("        → Create smaller subset for faster iteration")
    print("\n  Messification:")
    print("    - messify_m5_data(df, cache_dir=None, ...)")
    print("        → Simulate real-world data quality issues")
    print("\n  Inspection:")
    print("    - check_gaps(df, id_col='unique_id', date_col='ds', freq='W')")
    print("        → Diagnose gaps in time series data")
    print("    - first_contact_check(df, date_col='ds', target_col='y')")
    print("        → Run data quality checks")

    print("\n" + "=" * 70)
    print("Example usage:")
    print("=" * 70)
    print("""
from pathlib import Path
from m5_utils import load_m5

DATA_DIR = Path('data')

# Basic load (daily, unique_id)
df = load_m5(DATA_DIR)

# With hierarchy columns instead of unique_id
df = load_m5(DATA_DIR, include_hierarchy=True)
# Columns: item_id, dept_id, cat_id, store_id, state_id, ds, y

# Messified for training exercises
df = load_m5(DATA_DIR, messify=True, include_hierarchy=True)

# Custom messification options (new consistent naming!)
df = load_m5(
    DATA_DIR,
    messify=True,
    messify_kwargs={
        'zeros_to_na_frac': 0.20,        # 20% of zeros → NA
        'zeros_drop_frac': 0.02,          # Drop 2% of zero rows
        'zeros_drop_gaps_frac': 0.10,     # Drop 10% of middle zeros (gaps)
        'duplicates_add_n': 200,          # Add 200 duplicates
        'na_drop_frac': None,             # Don't drop NAs
        'dtypes_corrupt': True,           # Corrupt dtypes
        'cache_dir': DATA_DIR
    },
    include_hierarchy=True
)
    """)
    print("=" * 70)

__all__ = [
    # Loading functions
    'load_m5',
    'load_m5_with_feedback',
    'load_m5_calendar',
    'has_m5_cache',
    # Preprocessing
    'create_subset',
    'create_unique_id',
    'messify_m5_data',
    'expand_hierarchy',
    'aggregate_calendar_to_weekly',
    # Inspection
    'check_gaps',
    # Utilities
    'find_project_root',
    'get_notebook_name',
    'get_module_from_notebook',
    # Caching
    'CacheManager',
    'CacheEntry',
    # Constants
    'HIERARCHY_COLS',
]
