"""
Cache Manager
=============

CacheManager for storing and tracking datasets with:
- Automatic source/lineage tracking
- Config inheritance from parent datasets
- Cross-manager config lookup

ArtifactManager for curriculum outputs:
- Separate output/ directory
- Module-level manifest tracking
- Reports regenerated on load (no report files stored)
"""

import json
import hashlib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from ..utils.helpers import (
    get_notebook_name,
    get_module_from_notebook,
    get_artifact_subfolder,
)

if TYPE_CHECKING:
    from ..reports import ModuleReport


MANIFEST_FILENAME = 'cache_manifest.json'

# Module-level tracking (shared across all CacheManager instances)
_load_history: List[str] = []
_all_managers: List['CacheManager'] = []


@dataclass
class CacheEntry:
    """Metadata for a cached dataset."""
    key: str
    filename: str
    module: str
    config: Dict[str, Any]
    config_hash: str
    created_at: str
    rows: int
    columns: List[str]
    size_mb: float
    source: Optional[str] = None


class CacheManager:
    """
    Manages cached datasets with automatic lineage and config inheritance.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cached files
    overwrite_existing : bool, default=True
        Default behavior when saving to an existing key.
        
    Examples
    --------
    >>> cache = CacheManager(Path('data/cache'))
    >>> outputs = CacheManager(Path('data/outputs'))
    >>> 
    >>> # Save with auto-detected source and inherited config
    >>> outputs.save(df, config={'freq': 'W'})
    >>> 
    >>> # Load with report (regenerated on demand)
    >>> df, report = outputs.load('1_06_output', with_report=True)
    """
    
    def __init__(self, cache_dir: Path, overwrite_existing: bool = True):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / MANIFEST_FILENAME
        self._manifest = self._load_manifest()
        self.overwrite_existing = overwrite_existing
        _all_managers.append(self)
    
    def _load_manifest(self) -> Dict[str, dict]:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2, default=str)
    
    @staticmethod
    def _hash_config(config: dict) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    @staticmethod
    def last_loaded() -> Optional[str]:
        """Return the most recently loaded cache key."""
        return _load_history[-1] if _load_history else None
    
    @staticmethod
    def load_history() -> List[str]:
        """Return full load history for this session."""
        return _load_history.copy()
    
    @staticmethod
    def clear_history():
        """Clear load history."""
        _load_history.clear()
    
    @staticmethod
    def get_source_config(source_key: str) -> Optional[Dict[str, Any]]:
        """Look up config for a source key across all managers."""
        for manager in _all_managers:
            if source_key in manager._manifest:
                return manager._manifest[source_key].get('config', {}).copy()
        return None
    
    def save(
        self,
        df: pd.DataFrame,
        key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        module: Optional[str] = None,
        source: Optional[str] = None,
        inherit_config: bool = True,
        overwrite: Optional[bool] = None
    ) -> Path:
        """
        Save DataFrame with automatic source and config inheritance.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        key : str, optional
            Cache key. Defaults to '{notebook_name}_output'
        config : dict, optional
            Additional config (merged with inherited config)
        module : str, optional
            Module identifier. Auto-detects from notebook.
        source : str, optional
            Parent cache key. Auto-detects from last load.
        inherit_config : bool, default=True
            If True, inherit config from source
        overwrite : bool, optional
            Whether to overwrite. Defaults to overwrite_existing setting.
            
        Returns
        -------
        Path
            Path to saved data file
        """
        # Resolve overwrite setting
        if overwrite is None:
            overwrite = self.overwrite_existing
            
        # Auto-detect key
        if key is None:
            nb_name = get_notebook_name()
            if nb_name:
                key = f"{nb_name}_output"
            else:
                raise ValueError("Could not auto-detect key. Provide explicitly.")
        
        # Auto-detect module
        if module is None:
            module = get_module_from_notebook() or 'unknown'
        
        # Auto-detect source from load history
        if source is None:
            source = self.last_loaded()
        
        # Build config: inherit from source + merge overrides
        final_config = {}
        inherited_keys = 0
        
        if inherit_config and source:
            source_config = self.get_source_config(source)
            if source_config:
                final_config = source_config
                inherited_keys = len(source_config)
        
        # Merge provided config (overrides inherited values)
        if config:
            final_config.update(config)
        
        if key in self._manifest and not overwrite:
            print(f"⚠ Cache '{key}' exists. Use overwrite=True to replace.")
            return self.cache_dir / self._manifest[key]['filename']
        
        # Save data
        config_hash = self._hash_config(final_config)
        filename = f"{key}_{config_hash}.parquet"
        filepath = self.cache_dir / filename
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False)
        
        # Create manifest entry (no report file - regenerated on load)
        entry = CacheEntry(
            key=key,
            filename=filename,
            module=module,
            config=final_config,
            config_hash=config_hash,
            created_at=datetime.now().isoformat(),
            rows=len(df),
            columns=list(df.columns),
            size_mb=round(filepath.stat().st_size / 1024**2, 2),
            source=source,
        )
        
        self._manifest[key] = asdict(entry)
        self._save_manifest()
        
        # Output
        print(f"✓ Saved '{key}'")
        print(f"   Data:   {filename} ({entry.size_mb} MB, {entry.rows:,} rows)")
        if source:
            print(f"   Source: {source}")
        if inherited_keys:
            override_count = len(config) if config else 0
            print(f"   Config: {inherited_keys} inherited + {override_count} new")
        
        return filepath
    
    def load(
        self,
        key: str,
        config: Optional[Dict[str, Any]] = None,
        with_report: bool = False,
        verbose: bool = True
    ):
        """
        Load DataFrame from cache (and track for lineage).
        
        Parameters
        ----------
        key : str
            Cache key to load
        config : dict, optional
            Validates against stored config
        with_report : bool, default=False
            Return (df, report) tuple. Report is regenerated on demand.
        verbose : bool, default=True
            Print loading info
            
        Returns
        -------
        pd.DataFrame or tuple
            Data, or (data, ModuleReport) if with_report=True
        """
        if key not in self._manifest:
            if verbose:
                print(f"⚠ Cache '{key}' not found")
            return (None, None) if with_report else None
        
        entry = self._manifest[key]
        
        # Config validation
        if config is not None:
            current_hash = self._hash_config(config)
            if current_hash != entry['config_hash']:
                if verbose:
                    print(f"⚠ Cache '{key}' config mismatch - will regenerate")
                return (None, None) if with_report else None
        
        # Load data
        filepath = self.cache_dir / entry['filename']
        if not filepath.exists():
            if verbose:
                print(f"⚠ Cache file missing: {filepath}")
            return (None, None) if with_report else None
        
        df = pd.read_parquet(filepath)
        
        # Track this load for lineage
        _load_history.append(key)
        
        if verbose:
            print(f"✓ Loaded '{key}'")
            print(f"   Module: {entry['module']} | Shape: {entry['rows']:,} × {len(entry['columns'])}")
        
        # Regenerate report if requested
        if with_report:
            from ..reports import ModuleReport
            # Extract module ID from key (e.g., "1.06_first_contact" -> "1.06")
            module_id = entry.get('module', key.split('_')[0])
            report = ModuleReport(module=module_id, input_df=df)
            if verbose:
                print(f"   Report: ✓ (regenerated)")
            return df, report
        
        return df
    
    def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get config for a cached dataset."""
        if key in self._manifest:
            return self._manifest[key].get('config', {}).copy()
        return None
    
    def exists(self, key: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a cache key exists (optionally with matching config)."""
        if key not in self._manifest:
            return False
        if config is not None:
            return self._hash_config(config) == self._manifest[key]['config_hash']
        return True
    
    def info(self, key: str) -> Optional[dict]:
        """Print detailed info about a cached dataset."""
        if key not in self._manifest:
            print(f"⚠ Cache '{key}' not found")
            return None
        
        entry = self._manifest[key]
        print(f"\n{'='*60}")
        print(f"CACHE: {key}")
        print(f"{'='*60}")
        print(f"  Data:     {entry['filename']}")
        print(f"  Module:   {entry['module']}")
        print(f"  Created:  {entry['created_at'][:19]}")
        print(f"  Size:     {entry['size_mb']} MB")
        print(f"  Shape:    {entry['rows']:,} × {len(entry['columns'])}")
        if entry.get('source'):
            print(f"  Source:   {entry['source']}")
        print(f"\n  Config ({len(entry['config'])} keys):")
        for k, v in entry['config'].items():
            print(f"    {k}: {v}")
        print(f"{'='*60}\n")
        return entry
    
    def list(self) -> pd.DataFrame:
        """List all cached datasets."""
        if not self._manifest:
            print("📦 No cached datasets found.")
            return pd.DataFrame()
        
        rows = [{
            'Key': key,
            'Module': e['module'],
            'Rows': f"{e['rows']:,}",
            'Size (MB)': e['size_mb'],
            'Source': e.get('source') or '-'
        } for key, e in self._manifest.items()]
        
        df = pd.DataFrame(rows)
        print(f"\n📦 Cached Datasets ({len(df)}):\n")
        print(df.to_string(index=False))
        return df
    
    def lineage(self, key: str) -> list:
        """Show data lineage for a cached dataset."""
        if key not in self._manifest:
            print(f"⚠ Cache '{key}' not found")
            return []
        
        chain = [key]
        current = key
        
        while True:
            entry = None
            for manager in _all_managers:
                if current in manager._manifest:
                    entry = manager._manifest[current]
                    break
            
            if not entry or not entry.get('source'):
                break
            chain.append(entry['source'])
            current = entry['source']
        
        print(f"\n📜 Lineage for '{key}':")
        for i, item in enumerate(reversed(chain)):
            indent = "  " * i
            arrow = "→ " if i > 0 else ""
            
            module = '?'
            for manager in _all_managers:
                if item in manager._manifest:
                    module = manager._manifest[item].get('module', '?')
                    break
            
            print(f"   {indent}{arrow}{item} ({module})")
        
        return list(reversed(chain))
    
    def delete(self, key: str):
        """Delete a cached dataset."""
        if key not in self._manifest:
            print(f"⚠ Cache '{key}' not found")
            return
        
        entry = self._manifest[key]
        
        filepath = self.cache_dir / entry['filename']
        if filepath.exists():
            filepath.unlink()
        
        del self._manifest[key]
        self._save_manifest()
        print(f"✓ Deleted '{key}'")
    
    def clear(self, confirm: bool = False):
        """Delete all cached datasets."""
        if not confirm:
            print("⚠ Use clear(confirm=True) to delete all cached data.")
            return
        for key in list(self._manifest.keys()):
            self.delete(key)
        print("✓ Cache cleared")


class ArtifactManager:
    """
    Manages curriculum artifacts with separate data directory.

    Reports are regenerated on load - no report files stored.

    Structure:
        output/
        ├── data/
        │   └── 1_06_first_contact_output.parquet
        └── manifest.json

    Parameters
    ----------
    outputs_dir : Path
        Root output directory (e.g., DATA_DIR / 'output')

    Examples
    --------
    >>> artifacts = ArtifactManager(DATA_DIR / 'output')
    >>> artifacts.save(df)  # Auto-detects notebook name
    >>> df, report = artifacts.load('1.06_first_contact', with_report=True)
    """

    def __init__(self, outputs_dir: Path):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / 'data').mkdir(exist_ok=True)
        self.manifest_path = self.outputs_dir / 'manifest.json'
        self._manifest = self._load_json(self.manifest_path)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_json(path: Path, data: dict):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save(
        self,
        df: pd.DataFrame,
        key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        report: Optional['ModuleReport'] = None,
    ) -> Path:
        """
        Save DataFrame to outputs.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        key : str, optional
            Artifact key (e.g., '1.06_first_contact'). Auto-detects from notebook.
        config : dict, optional
            Config metadata to store
        source : str, optional
            Source artifact key for lineage. Auto-detects from last load.
        report : ModuleReport, optional
            Report object (displayed but not saved - regenerated on load)

        Returns
        -------
        Path
            Path to saved parquet file
        """
        # Auto-detect key from notebook name
        if key is None:
            key = get_notebook_name()
            if key is None:
                raise ValueError("Could not auto-detect key. Provide explicitly.")

        # Auto-detect source from load history (shared with CacheManager)
        if source is None:
            source = _load_history[-1] if _load_history else None

        # Extract module from key for manifest
        module = key.split('_')[0] if '_' in key else key

        # Save data
        data_filename = f'{key}_output.parquet'
        data_path = self.outputs_dir / 'data' / data_filename
        df.to_parquet(data_path, index=False)

        # Update manifest (no report file - regenerated on load)
        self._manifest[key] = {
            'key': key,
            'module': module,
            'data_file': f'data/{data_filename}',
            'config': config or {},
            'source': source,
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
            'size_mb': round(data_path.stat().st_size / 1024**2, 2),
        }
        self._save_json(self.manifest_path, self._manifest)

        # Output
        print(f"✓ Saved '{key}'")
        print(f"   Data: data/{data_filename} ({self._manifest[key]['size_mb']} MB, {len(df):,} rows)")

        # Display report if provided (but don't save it)
        if report is not None:
            print(f"   Report: displayed (regenerate on load with with_report=True)")

        return data_path

    def load(
        self,
        key: str,
        with_report: bool = False,
    ):
        """
        Load DataFrame (and optional report) from outputs.

        Parameters
        ----------
        key : str
            Artifact key (e.g., '1.06_first_contact')
        with_report : bool, default=False
            Return (df, report) tuple. Report is regenerated from data.

        Returns
        -------
        pd.DataFrame or tuple
            Data, or (data, report) if with_report=True
        """
        if key not in self._manifest:
            print(f"⚠ Artifact '{key}' not found")
            return (None, None) if with_report else None

        entry = self._manifest[key]

        # Load data
        data_path = self.outputs_dir / entry['data_file']
        if not data_path.exists():
            print(f"⚠ Data file missing: {data_path}")
            return (None, None) if with_report else None

        df = pd.read_parquet(data_path)
        
        # Track for lineage
        _load_history.append(key)
        
        print(f"✓ Loaded '{key}'")
        print(f"   Shape: {len(df):,} × {len(df.columns)}")

        if with_report:
            from ..reports import ModuleReport
            # Extract module ID (e.g., "1.06")
            module_id = entry.get('module', key.split('_')[0])
            report = ModuleReport(module=module_id, input_df=df)
            print(f"   Report: ✓ (regenerated)")
            return df, report

        return df

    def info(self, key: str) -> Optional[dict]:
        """Print detailed info about an artifact."""
        if key not in self._manifest:
            print(f"⚠ Artifact '{key}' not found")
            return None

        entry = self._manifest[key]
        print(f"\n{'='*60}")
        print(f"ARTIFACT: {key}")
        print(f"{'='*60}")
        print(f"  Data:     {entry['data_file']}")
        print(f"  Module:   {entry.get('module', 'unknown')}")
        print(f"  Created:  {entry['created_at'][:19]}")
        print(f"  Size:     {entry['size_mb']} MB")
        print(f"  Shape:    {entry['rows']:,} × {len(entry['columns'])}")
        if entry.get('source'):
            print(f"  Source:   {entry['source']}")
        if entry.get('config'):
            print(f"\n  Config:")
            for k, v in entry['config'].items():
                print(f"    {k}: {v}")
        print(f"{'='*60}\n")
        return entry

    def list(self) -> pd.DataFrame:
        """List all artifacts."""
        rows = [{
            'Key': key,
            'Module': e.get('module', '-'),
            'Rows': f"{e['rows']:,}",
            'Size (MB)': e['size_mb'],
        } for key, e in self._manifest.items()]

        if not rows:
            print("📦 No artifacts found.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"\n📦 Artifacts ({len(df)}):\n")
        print(df.to_string(index=False))
        return df


class NullCacheManager:
    """
    No-op cache used when use_cache=False.

    Behaves like CacheManager but:
    - load(...) always returns None
    - save(...) does nothing

    This allows notebook code to always call cache.load/save
    without conditionals.
    """

    def __init__(self, cache_dir: Path, *args, **kwargs):
        self.cache_dir = Path(cache_dir)

    def load(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def exists(self, *args, **kwargs):
        return False

    def list(self):
        return []

    def info(self, *args, **kwargs):
        return None

    def lineage(self, *args, **kwargs):
        return []

    def clear(self, *args, **kwargs):
        return None


__all__ = ['CacheManager', 'CacheEntry', 'ArtifactManager', 'NullCacheManager']