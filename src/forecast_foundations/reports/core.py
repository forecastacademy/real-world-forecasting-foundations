"""
Core report classes: ModuleReport, Snapshot.

ModuleReport provides a consistent interface across all modules:
    report.summary      # dict
    report.target       # list[check] - Q1
    report.metric       # list[check] - Q2
    report.structure    # list[check] - Q3
    report.drivers      # list[check] - Q4
    report.readiness    # list[check] - final status (optional)
    report.decisions    # str
    report.changes      # dict (only if output_df provided)

Reports are regenerated on demand from data - no save/load needed.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from .checks import MODULE_CHECKS, MODULE_TITLES
from .formatters import (
    format_table,
    format_decisions_df,
    decisions_df_to_markdown,
    render_checks_text,
    render_checks_markdown,
    section_header,
    get_memory_tier,
)


# =============================================================================
# DECISIONS LOADER
# =============================================================================

def _find_project_root(start_path: Path = None) -> Optional[Path]:
    """Find project root by looking for config/decisions.yaml or pyproject.toml."""
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    for _ in range(10):
        if (current / "config" / "decisions.yaml").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _load_decisions_from_yaml(module: str, root: Path = None) -> Optional[pd.DataFrame]:
    """Load decisions for a module from config/decisions.yaml."""
    if root is None:
        root = _find_project_root()
    
    if root is None:
        return None
    
    decisions_path = root / "config" / "decisions.yaml"
    if not decisions_path.exists():
        return None
    
    try:
        with open(decisions_path) as f:
            data = yaml.safe_load(f)
        
        if module not in data:
            return None
        
        module_data = data[module]
        decisions_list = module_data.get("decisions", [])
        
        if not decisions_list:
            return None
        
        return pd.DataFrame(decisions_list)
    except Exception:
        return None


# =============================================================================
# SNAPSHOT
# =============================================================================

@dataclass
class Snapshot:
    """Data state capture."""
    
    name: str
    rows: int
    columns: int
    series: int
    date_min: str
    date_max: str
    n_weeks: int
    frequency: str
    target_zeros_pct: float
    target_nas: int
    duplicates: int
    memory_mb: float
    
    @classmethod
    def from_df(
        cls, 
        df: pd.DataFrame, 
        name: str = "data",
        date_col: str = 'ds',
        target_col: str = 'y',
        id_col: str = 'unique_id'
    ) -> 'Snapshot':
        rows = len(df)
        columns = df.shape[1]
        series = df[id_col].nunique() if id_col in df.columns else 0
        
        date_min = date_max = 'N/A'
        n_weeks = 0
        frequency = 'N/A'
        
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            min_d, max_d = df[date_col].min(), df[date_col].max()
            date_min = str(min_d.date())
            date_max = str(max_d.date())
            n_weeks = int((max_d - min_d).days / 7)
            
            unique_dates = df[date_col].drop_duplicates().sort_values()
            if len(unique_dates) > 1:
                freq_days = unique_dates.diff().mode().iloc[0].days
                freq_map = {1: 'Daily', 7: 'Weekly', 14: 'Biweekly', 30: 'Monthly', 31: 'Monthly'}
                frequency = freq_map.get(freq_days, f'{freq_days} days')
        
        target_zeros_pct = 0.0
        target_nas = 0
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_zeros_pct = (df[target_col] == 0).mean() * 100
            target_nas = int(df[target_col].isna().sum())
        
        key_cols = [c for c in [date_col, id_col] if c in df.columns]
        duplicates = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        return cls(
            name=name, rows=rows, columns=columns, series=series,
            date_min=date_min, date_max=date_max, n_weeks=n_weeks,
            frequency=frequency, target_zeros_pct=target_zeros_pct,
            target_nas=target_nas, duplicates=duplicates, memory_mb=memory_mb
        )


# =============================================================================
# MODULE REPORT
# =============================================================================

class ModuleReport:
    """
    Unified report class for all modules.
    
    Provides consistent properties across all modules:
        report.summary, report.target, report.metric, report.structure,
        report.drivers, report.readiness, report.decisions, report.changes
    
    Reports are regenerated on demand - no save/load needed.
    
    Parameters
    ----------
    module : str
        Module ID (e.g., "1.06", "1.08")
    input_df : pd.DataFrame
        Data to report on (or data before transformations if output_df provided)
    output_df : pd.DataFrame, optional
        Data after transformations. If None, report shows only input_df state
        and CHANGES section is hidden.
    decisions : str, pd.DataFrame, or None
        Explicit decisions. If None, auto-loads from config/decisions.yaml.
    drivers : dict[str, pd.DataFrame], optional
        Driver datasets (calendar, prices) for validation.
        
    Examples
    --------
    >>> # Single DataFrame - snapshot report (no changes)
    >>> report = ModuleReport("1.06", df)
    >>> report.display()
    
    >>> # Before/after - shows changes
    >>> report = ModuleReport("1.08", input_df=raw_df, output_df=clean_df)
    >>> report.display()
    """
    
    def __init__(
        self,
        module: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame = None,
        decisions: Union[str, pd.DataFrame, None] = None,
        drivers: Dict[str, pd.DataFrame] = None,
        date_col: str = 'ds',
        target_col: str = 'y',
        id_col: str = 'unique_id',
        sample_df: pd.DataFrame = None,
        hierarchy_cols: List[str] = None,
        **kwargs
    ):
        self.module = module
        self.module_title = MODULE_TITLES.get(module, "")
        
        # If no output_df, report on input_df only (no changes section)
        if output_df is None:
            output_df = input_df
            self._show_changes = False
        else:
            self._show_changes = True
        
        self.input = Snapshot.from_df(input_df, "Input", date_col, target_col, id_col)
        self.output = Snapshot.from_df(output_df, "Output", date_col, target_col, id_col)
        self.sample_df = sample_df if sample_df is not None else output_df
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Load decisions
        if decisions is not None:
            if isinstance(decisions, pd.DataFrame):
                self.decisions_df = decisions
                self._decisions_md = None
            else:
                self.decisions_df = None
                self._decisions_md = decisions.strip()
        else:
            self.decisions_df = _load_decisions_from_yaml(module)
            self._decisions_md = None
        
        # Params for check functions
        if hierarchy_cols is None:
            hierarchy_cols = ['state_id', 'store_id', 'cat_id', 'dept_id']
        
        params = {
            'date_col': date_col,
            'target_col': target_col,
            'id_col': id_col,
            'hierarchy_cols': hierarchy_cols,
            'drivers': drivers or {},
            'input_df': input_df,
            **kwargs
        }
        
        # Run module-specific checks
        module_checks = MODULE_CHECKS.get(module, {})
        self._target = module_checks.get('target', lambda *a, **k: [])(output_df, **params)
        self._metric = module_checks.get('metric', lambda *a, **k: [])(output_df, **params)
        self._structure = module_checks.get('structure', lambda *a, **k: [])(output_df, **params)
        self._drivers = module_checks.get('drivers', lambda *a, **k: [])(output_df, **params)
        self._readiness = module_checks.get('readiness', lambda *a, **k: [])(output_df, **params)
    
    # -------------------------------------------------------------------------
    # Properties - Consistent API
    # -------------------------------------------------------------------------
    
    @property
    def summary(self) -> dict:
        """DATA SUMMARY as dict."""
        return {
            'Rows': f"{self.output.rows:,}",
            'Series': f"{self.output.series:,}",
            'Dates': f"{self.output.date_min} → {self.output.date_max}",
            'Frequency': self.output.frequency,
            'History': f"{self.output.n_weeks} weeks ({self.output.n_weeks/52:.1f} yrs)",
            'Target zeros': f"{self.output.target_zeros_pct:.1f}%",
        }
    
    @property
    def target(self) -> List[Dict]:
        """Q1: TARGET checks."""
        return self._target
    
    @property
    def metric(self) -> List[Dict]:
        """Q2: METRIC checks."""
        return self._metric
    
    @property
    def structure(self) -> List[Dict]:
        """Q3: STRUCTURE checks."""
        return self._structure
    
    @property
    def drivers(self) -> List[Dict]:
        """Q4: DRIVERS checks."""
        return self._drivers
    
    @property
    def readiness(self) -> List[Dict]:
        """READINESS checks (prep modules only)."""
        return self._readiness
    
    @property
    def decisions(self) -> str:
        """DECISIONS as formatted string."""
        if self.decisions_df is not None and not self.decisions_df.empty:
            return format_decisions_df(self.decisions_df)
        if self._decisions_md:
            return self._decisions_md
        return ""
    
    @property
    def changes(self) -> dict:
        """CHANGES vs input. Empty dict if no output_df was provided."""
        if not self._show_changes:
            return {}
        
        row_delta = self.output.rows - self.input.rows
        row_pct = (row_delta / self.input.rows * 100) if self.input.rows > 0 else 0
        result = {
            'rows': {'before': self.input.rows, 'after': self.output.rows, 'pct': round(row_pct, 0)},
        }
        if self.input.target_nas > 0 or self.output.target_nas > 0:
            result['nas'] = {'before': self.input.target_nas, 'after': self.output.target_nas}
        if self.input.frequency != self.output.frequency:
            result['frequency'] = {'before': self.input.frequency, 'after': self.output.frequency}
        if self.input.memory_mb > 0:
            mem_pct = ((self.input.memory_mb - self.output.memory_mb) / self.input.memory_mb * 100)
            result['memory'] = {
                'before_mb': round(self.input.memory_mb, 1),
                'after_mb': round(self.output.memory_mb, 1),
                'pct': round(mem_pct, 0)
            }
        return result
    
    @property
    def blocking_issues(self) -> List[str]:
        """List of blocking issues (status == '✗')."""
        issues = []
        for checks in [self._target, self._metric, self._structure, self._drivers]:
            for c in checks:
                if c.get('status') == '✗':
                    issues.append(f"{c['check']}: {c['value']}")
        return issues
    
    @property
    def memory(self) -> dict:
        """Memory assessment."""
        tier, note, status = get_memory_tier(self.output.memory_mb)
        return {
            'size_mb': round(self.output.memory_mb, 1),
            'tier': tier,
            'note': note,
            'status': status,
        }
    
    # -------------------------------------------------------------------------
    # Output Methods
    # -------------------------------------------------------------------------
    
    def to_text(self) -> str:
        """Generate text report."""
        w = 65
        lines = []
        
        # Header
        title = f"{self.module} · {self.module_title}" if self.module_title else self.module
        lines.append("━" * w)
        lines.append(title)
        lines.append("━" * w)
        
        # Snapshot
        lines.append(section_header("SNAPSHOT"))
        lines.append(self.sample_df.head(3).to_string(index=False))
        
        # Data Summary
        lines.append(section_header("DATA SUMMARY"))
        for k, v in self.summary.items():
            lines.append(f"  {k:<15} {v}")
        
        # Memory
        lines.append(section_header("MEMORY"))
        mem = self.memory
        lines.append(f"  {mem['status']} {mem['size_mb']:.0f} MB ({mem['tier']}) — {mem['note']}")
        
        # 5Q Sections
        q_sections = [
            ('Q1 · Target', self._target),
            ('Q2 · Metric', self._metric),
            ('Q3 · Structure', self._structure),
            ('Q4 · Drivers', self._drivers),
        ]
        
        has_checks = any(checks for _, checks in q_sections)
        if has_checks:
            lines.append(section_header("5Q CHECKS"))
            for name, checks in q_sections:
                if checks:
                    lines.append(f"\n  {name}")
                    lines.extend(render_checks_text(checks, indent="    "))
        
        # Readiness (if present)
        if self._readiness:
            lines.append(section_header("READINESS"))
            lines.extend(render_checks_text(self._readiness))
        
        # Blocking Issues
        if has_checks:
            lines.append(section_header("BLOCKING ISSUES"))
            if self.blocking_issues:
                for b in self.blocking_issues:
                    lines.append(f"  ✗ {b}")
            else:
                lines.append("  None ✓")
        
        # Decisions
        if self.decisions:
            lines.append(section_header("DECISIONS"))
            lines.append(self.decisions)
        
        # Changes (only if output_df was provided)
        if self._show_changes:
            lines.append(section_header("CHANGES"))
            c = self.changes
            row_pct = c['rows']['pct']
            sign = '+' if row_pct >= 0 else ''
            lines.append(f"  {'Rows':<15} {c['rows']['before']:,} → {c['rows']['after']:,}  ({sign}{row_pct:.0f}%)")
            if 'nas' in c:
                fixed = "✓ Fixed" if c['nas']['after'] == 0 and c['nas']['before'] > 0 else ""
                lines.append(f"  {'NAs (y)':<15} {c['nas']['before']:,} → {c['nas']['after']:,}  {fixed}")
            if 'frequency' in c:
                lines.append(f"  {'Frequency':<15} {c['frequency']['before']} → {c['frequency']['after']}")
            if 'memory' in c:
                mem_sign = '+' if c['memory']['pct'] < 0 else '-'
                lines.append(f"  {'Memory':<15} {c['memory']['before_mb']:.1f} MB → {c['memory']['after_mb']:.1f} MB  ({mem_sign}{abs(c['memory']['pct']):.0f}%)")
        
        # Footer
        lines.append("\n" + "━" * w)
        lines.append(f"Generated: {self.generated_at}")
        lines.append("━" * w)
        
        return '\n'.join(lines)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        
        # Header
        title = f"{self.module} · {self.module_title}" if self.module_title else self.module
        lines.append(f"# {title}\n")
        
        # Summary
        lines.append("## Data Summary\n")
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")
        
        # Memory
        mem = self.memory
        lines.append(f"\n**Memory:** {mem['status']} {mem['size_mb']:.0f} MB ({mem['tier']}) — {mem['note']}\n")
        
        # 5Q Sections
        q_sections = [
            ('Q1: Target', self._target),
            ('Q2: Metric', self._metric),
            ('Q3: Structure', self._structure),
            ('Q4: Drivers', self._drivers),
        ]
        
        for name, checks in q_sections:
            if checks:
                lines.append(f"\n## {name}\n")
                lines.extend(render_checks_markdown(checks))
        
        # Readiness
        if self._readiness:
            lines.append("\n## Readiness\n")
            lines.extend(render_checks_markdown(self._readiness))
        
        # Blocking Issues
        lines.append("\n## Blocking Issues\n")
        if self.blocking_issues:
            for b in self.blocking_issues:
                lines.append(f"- ❌ {b}")
        else:
            lines.append("None ✓")
        
        # Decisions
        if self.decisions_df is not None and not self.decisions_df.empty:
            lines.append("\n## Decisions\n")
            lines.append(decisions_df_to_markdown(self.decisions_df))
        elif self._decisions_md:
            lines.append("\n## Decisions\n")
            lines.append(self._decisions_md)
        
        # Changes (only if output_df was provided)
        if self._show_changes:
            lines.append("\n## Changes\n")
            lines.append("| Metric | Before | After | Δ |")
            lines.append("|--------|--------|-------|---|")
            c = self.changes
            lines.append(f"| Rows | {c['rows']['before']:,} | {c['rows']['after']:,} | {c['rows']['pct']:+.0f}% |")
            if 'nas' in c:
                fixed = " ✓" if c['nas']['after'] == 0 else ""
                lines.append(f"| NAs (y) | {c['nas']['before']:,} | {c['nas']['after']:,} | Fixed{fixed} |")
            if 'frequency' in c:
                lines.append(f"| Frequency | {c['frequency']['before']} | {c['frequency']['after']} | — |")
            if 'memory' in c:
                lines.append(f"| Memory | {c['memory']['before_mb']:.0f} MB | {c['memory']['after_mb']:.0f} MB | {c['memory']['pct']:+.0f}% |")
        
        lines.append(f"\n---\n*Generated: {self.generated_at}*")
        
        return '\n'.join(lines)
    
    def display(self):
        """Print text report."""
        print(self.to_text())
    
    def __repr__(self):
        return f"ModuleReport({self.module}, {self.output.rows:,} rows, {self.output.series:,} series)"


# =============================================================================
# PLOT FUNCTION
# =============================================================================

def plot_timeline_health(
    df: pd.DataFrame,
    date_col: str = 'ds',
    id_col: str = 'unique_id',
    figsize: tuple = (12, 3),
    title: str = 'Timeline Health: Series Reporting per Date'
):
    """Single plot: series count per date. Flat = good."""
    import matplotlib.pyplot as plt
    
    series_per_date = df.groupby(date_col)[id_col].nunique()
    
    fig, ax = plt.subplots(figsize=figsize)
    series_per_date.plot(ax=ax, color='#2596be', linewidth=1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Series count')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(series_per_date.max(), color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig, ax


__all__ = ['ModuleReport', 'Snapshot', 'plot_timeline_health']