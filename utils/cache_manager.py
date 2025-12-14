"""
CacheManager with automatic source + config inheritance
========================================================

Features:
- Auto-detects source from load history
- Auto-inherits config from source
- Tracks reports alongside data
- Cross-manager config lookup (cache <-> outputs)
"""

import json
import hashlib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.report import FirstContactReport


MANIFEST_FILENAME = 'cache_manifest.json'

# Module-level tracking (shared across all CacheManager instances)
_load_history: List[str] = []
_all_managers: List['CacheManager'] = []


def get_notebook_name() -> Optional[str]:
    """Detect current notebook name."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython and hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            return Path(ipython.user_ns['__vsc_ipynb_file__']).stem
    except Exception:
        pass
    return None


def get_module_from_notebook() -> Optional[str]:
    """Extract module (first 4 chars) from notebook name."""
    nb_name = get_notebook_name()
    return nb_name[:4] if nb_name and len(nb_name) >= 4 else None


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
    report_filename: Optional[str] = None


class CacheManager:
    """
    Manages cached datasets with automatic lineage and config inheritance.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cached files
    overwrite_existing : bool, default=True
        Default behavior when saving to an existing key.
        - True: Overwrite existing cache without warning
        - False: Skip save if key exists (print warning)
        Can be overridden per-call with save(..., overwrite=True/False)
    """
    
    def __init__(self, cache_dir: Path, overwrite_existing: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        report: Optional['FirstContactReport'] = None,
        inherit_config: bool = True,
        overwrite: Optional[bool] = None  # None = use overwrite_existing
    ) -> Path:
        """
        Save DataFrame with automatic source and config inheritance.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing cache. If None, uses overwrite_existing
            from CacheManager initialization.
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
        df.to_parquet(filepath, index=False)
        
        # Save report if provided
        report_filename = None
        if report is not None:
            report_filename = f"{key}_report.json"
            report_path = self.cache_dir / report_filename
            report.save(report_path)
        
        # Create manifest entry
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
            report_filename=report_filename
        )
        
        self._manifest[key] = asdict(entry)
        self._save_manifest()
        
        # Output
        print(f"✓ Saved '{key}'")
        print(f"   Data:   {filename} ({entry.size_mb} MB, {entry.rows:,} rows)")
        if report_filename:
            print(f"   Report: {report_filename}")
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
        """Load DataFrame from cache (and track for lineage)."""
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
        
        # Load report if requested
        if with_report:
            report_obj = None
            if entry.get('report_filename'):
                report_path = self.cache_dir / entry['report_filename']
                if report_path.exists():
                    from utils.report import FirstContactReport
                    report_obj = FirstContactReport.load(report_path)
                    if verbose:
                        print(f"   Report: ✓")
            return df, report_obj
        
        return df
    
    def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get config for a cached dataset."""
        if key in self._manifest:
            return self._manifest[key].get('config', {}).copy()
        return None
    
    def exists(self, key: str, config: Optional[Dict[str, Any]] = None) -> bool:
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
        print(f"  Report:   {entry.get('report_filename') or 'None'}")
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
            'Report': '✓' if e.get('report_filename') else '-',
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
            has_report = False
            for manager in _all_managers:
                if item in manager._manifest:
                    module = manager._manifest[item].get('module', '?')
                    has_report = bool(manager._manifest[item].get('report_filename'))
                    break
            
            report_icon = ' 📋' if has_report else ''
            print(f"   {indent}{arrow}{item} ({module}){report_icon}")
        
        return list(reversed(chain))
    
    def delete(self, key: str):
        """Delete a cached dataset and its report."""
        if key not in self._manifest:
            print(f"⚠ Cache '{key}' not found")
            return
        
        entry = self._manifest[key]
        
        filepath = self.cache_dir / entry['filename']
        if filepath.exists():
            filepath.unlink()
        
        if entry.get('report_filename'):
            report_path = self.cache_dir / entry['report_filename']
            if report_path.exists():
                report_path.unlink()
        
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


__all__ = ['CacheManager', 'CacheEntry', 'get_notebook_name', 'get_module_from_notebook']