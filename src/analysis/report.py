"""
First Contact Report
====================

Data quality assessment tool that generates a structured "report card"
for time series datasets.

Features:
- Schema validation (columns, dtypes)
- Completeness checks (NAs)
- Validity checks (date ranges, negative values)
- Integrity checks (duplicates, frequency)
- Summary statistics
- Comparison between reports
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FirstContactReport:
    """
    Structured report card from first contact with a dataset.
    
    Attributes
    ----------
    checks : pd.DataFrame
        Individual check results with columns: Category, Check, Status, Value, Notes
    summary : dict
        Summary statistics (rows, columns, series count, etc.)
    dataset_name : str
        Name for display purposes
    generated_at : str
        Timestamp when report was generated
        
    Examples
    --------
    >>> report = first_contact_check(df, dataset_name='M5 Sales')
    >>> report.table()              # Styled table in notebook
    >>> report.blocking_issues()    # Just failures
    >>> report.save('report.json')  # Save for later
    >>> loaded = FirstContactReport.load('report.json')
    """
    checks: pd.DataFrame
    summary: dict
    dataset_name: str = "dataset"
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    def __repr__(self):
        passed = (self.checks['Status'] == '✓').sum()
        failed = (self.checks['Status'] == '✗').sum()
        return f"FirstContactReport('{self.dataset_name}': {passed} passed, {failed} failed)"
    
    def _repr_html_(self):
        """Auto-display styled table in Jupyter notebooks."""
        return self.table()._repr_html_()
    
    def table(self):
        """
        Return a styled report card table for notebook display.
        Color-coded by status for easy scanning.
        """
        display_df = self.checks.copy()
        
        def style_row(row):
            """Color entire row based on status."""
            colors = {
                '✓': 'background-color: #d4edda',  # Green - pass
                '✗': 'background-color: #f8d7da',  # Red - fail  
                '⚠': 'background-color: #fff3cd',  # Yellow - warning
                'ℹ': 'background-color: #e2e3e5',  # Gray - info
            }
            color = colors.get(row['Status'], '')
            return [color] * len(row)
        
        styled = (display_df.style
            .apply(style_row, axis=1)
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left'), ('font-weight', 'bold')]},
                {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold')]},
            ])
            .set_caption(f"📋 First Contact Report: {self.dataset_name} ({self.generated_at})")
            .hide(axis='index')
        )
        return styled
    
    def summary_table(self):
        """Return summary stats as a styled DataFrame."""
        summary_df = pd.DataFrame([
            {'Metric': k, 'Value': v} for k, v in self.summary.items()
        ])
        return (summary_df.style
            .set_properties(**{'text-align': 'left'})
            .set_caption(f"📊 Summary Stats: {self.dataset_name}")
            .hide(axis='index')
        )
    
    def display(self):
        """Print text version of report (for non-notebook use)."""
        print("\n" + "=" * 70)
        print(f"📋 FIRST CONTACT REPORT: {self.dataset_name}")
        print(f"   Generated: {self.generated_at}")
        print("=" * 70)
        
        for category in self.checks['Category'].unique():
            cat_checks = self.checks[self.checks['Category'] == category]
            print(f"\n{category}")
            print("-" * 70)
            for _, row in cat_checks.iterrows():
                note = f" → {row['Notes']}" if row['Notes'] else ""
                print(f"  {row['Status']} {row['Check']:<32} {str(row['Value']):<12}{note}")
        
        print(f"\n📊 SUMMARY")
        print("-" * 70)
        for key, val in self.summary.items():
            print(f"  {key:<25} {val}")
        
        failed = (self.checks['Status'] == '✗').sum()
        warnings = (self.checks['Status'] == '⚠').sum()
        print("\n" + "=" * 70)
        if failed > 0:
            print(f"❌ {failed} BLOCKING ISSUE(S)")
        elif warnings > 0:
            print(f"⚠️  {warnings} WARNING(S)")
        else:
            print("✅ ALL CHECKS PASSED")
        print("=" * 70)
        return self
    
    def blocking_issues(self) -> pd.DataFrame:
        """Return only failing checks."""
        return self.checks[self.checks['Status'] == '✗'].copy()
    
    def save(self, path: str):
        """
        Save report to JSON (preserves checks + summary).
        
        Parameters
        ----------
        path : str or Path
            Output path (.json recommended)
        """
        path = Path(path)
        data = {
            'dataset_name': self.dataset_name,
            'generated_at': self.generated_at,
            'summary': self.summary,
            'checks': self.checks.to_dict(orient='records')
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved report to {path.name}")
    
    @classmethod
    def load(cls, path: str) -> 'FirstContactReport':
        """
        Load report from JSON.
        
        Parameters
        ----------
        path : str or Path
            Path to saved report
            
        Returns
        -------
        FirstContactReport
            Reconstructed report object
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            checks=pd.DataFrame(data['checks']),
            summary=data['summary'],
            dataset_name=data['dataset_name'],
            generated_at=data['generated_at']
        )
    
    def compare(self, other: 'FirstContactReport') -> pd.DataFrame:
        """
        Compare two reports to show progress.
        
        Parameters
        ----------
        other : FirstContactReport
            Report to compare against (typically "after" state)
            
        Returns
        -------
        pd.DataFrame
            Comparison showing before/after status and what changed
        """
        comp = self.checks[['Check', 'Status', 'Value']].merge(
            other.checks[['Check', 'Status', 'Value']],
            on='Check', suffixes=('_before', '_after')
        )
        comp['Changed'] = comp['Status_before'] != comp['Status_after']
        return comp


def first_contact_check(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    dataset_name: str = 'dataset'
) -> FirstContactReport:
    """
    Run first-contact checks and return a structured report card.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    date_col : str
        Name of the date column
    target_col : str
        Name of the target column
    dataset_name : str
        Name for the report
    
    Returns
    -------
    FirstContactReport
        Use .table() for styled output, .checks for raw DataFrame
    
    Examples
    --------
    >>> report = first_contact_check(df, dataset_name='M5 Sales')
    >>> report.table()         # Styled table in notebook
    >>> report.summary_table() # Summary stats table
    >>> report.blocking_issues()  # Just failures
    >>> report.save('report.json')  # Save for later
    >>> loaded = FirstContactReport.load('report.json')  # Load back
    """
    
    checks = []
    id_cols = [c for c in df.columns if c not in [date_col, target_col]]
    n_rows = len(df)
    
    def add(category, check, status, value, notes=""):
        checks.append({
            'Category': category,
            'Check': check,
            'Status': status,
            'Value': value,
            'Notes': notes
        })
    
    # =========================================================================
    # 1. SCHEMA
    # =========================================================================
    has_date = date_col in df.columns
    has_target = target_col in df.columns
    
    add('Schema', f'Column: {date_col}', 
        '✓' if has_date else '✗', 
        'Present' if has_date else 'Missing',
        'Required')
    add('Schema', f'Column: {target_col}', 
        '✓' if has_target else '✗',
        'Present' if has_target else 'Missing',
        'Required')
    
    if not (has_date and has_target):
        return FirstContactReport(
            checks=pd.DataFrame(checks),
            summary={'Error': 'Missing required columns'},
            dataset_name=dataset_name
        )
    
    date_is_dt = pd.api.types.is_datetime64_any_dtype(df[date_col])
    target_is_num = pd.api.types.is_numeric_dtype(df[target_col])
    
    add('Schema', f'Type: {date_col}',
        '✓' if date_is_dt else '✗',
        str(df[date_col].dtype),
        '' if date_is_dt else 'Use pd.to_datetime()')
    add('Schema', f'Type: {target_col}',
        '✓' if target_is_num else '✗',
        str(df[target_col].dtype),
        '' if target_is_num else 'Convert to numeric')
    
    # =========================================================================
    # 2. COMPLETENESS
    # =========================================================================
    na_date = df[date_col].isna().sum()
    na_target = df[target_col].isna().sum()
    na_pct = na_target / n_rows * 100 if n_rows > 0 else 0
    
    add('Completeness', f'NAs: {date_col}',
        '✓' if na_date == 0 else '✗',
        f'{na_date:,}',
        '' if na_date == 0 else 'Dates cannot be null')
    add('Completeness', f'NAs: {target_col}',
        '✓' if na_target == 0 else 'ℹ',
        f'{na_target:,} ({na_pct:.1f}%)',
        '' if na_target == 0 else 'Handle in imputation')
    
    if id_cols:
        na_ids = df[id_cols].isna().any(axis=1).sum()
        add('Completeness', 'NAs: ID columns',
            '✓' if na_ids == 0 else '✗',
            f'{na_ids:,}',
            '' if na_ids == 0 else 'IDs cannot be null')
    
    # =========================================================================
    # 3. VALIDITY
    # =========================================================================
    if date_is_dt:
        dates = df[date_col]
        min_date, max_date = dates.min(), dates.max()
        today = pd.Timestamp.today()
        
        n_old = (dates < '1900-01-01').sum()
        n_future = (dates > today).sum()
        
        add('Validity', 'Dates ≥ 1900',
            '✓' if n_old == 0 else '✗',
            f'{n_old:,} invalid',
            '' if n_old == 0 else 'Check for errors')
        add('Validity', 'No future dates',
            '✓' if n_future == 0 else '⚠',
            f'{n_future:,} future',
            '' if n_future == 0 else 'May need filtering')
    
    if target_is_num:
        target = df[target_col]
        n_neg = (target < 0).sum()
        add('Validity', f'{target_col} ≥ 0',
            'ℹ',
            f'{n_neg:,} negative',
            'Review if unexpected')
    
    # =========================================================================
    # 4. INTEGRITY
    # =========================================================================
    key_cols = [date_col] + id_cols
    n_dups = df.duplicated(subset=key_cols).sum()
    add('Integrity', 'No duplicate keys',
        '✓' if n_dups == 0 else '✗',
        f'{n_dups:,}',
        '' if n_dups == 0 else f'Keys: {", ".join(key_cols[:2])}...')
    
    if date_is_dt:
        unique_dates = df[date_col].drop_duplicates().sort_values()
        if len(unique_dates) > 1:
            freq = unique_dates.diff().mode().iloc[0]
            freq_str = str(freq).replace('0 days ', '').replace('00:00:00', '').strip()
            if not freq_str:
                freq_str = 'Daily'
            add('Integrity', 'Frequency',
                'ℹ',
                freq_str,
                'Check for gaps')
    
    # =========================================================================
    # SUMMARY STATS
    # =========================================================================
    summary = {
        'Rows': f'{n_rows:,}',
        'Columns': f'{df.shape[1]}',
    }
    
    if id_cols:
        if len(id_cols) == 1:
            n_series = df[id_cols[0]].nunique()
        else:
            n_series = df.groupby(id_cols, sort=False).ngroups
        summary['Series'] = f'{n_series:,}'
        summary['ID columns'] = ', '.join(id_cols)
    
    if date_is_dt:
        summary['Date range'] = f'{min_date.date()} → {max_date.date()}'
        summary['Unique dates'] = f'{df[date_col].nunique():,}'
    
    if target_is_num:
        summary[f'{target_col} mean'] = f'{df[target_col].mean():,.2f}'
        n_zeros = (df[target_col] == 0).sum()
        summary[f'{target_col} zeros'] = f'{n_zeros:,} ({n_zeros/n_rows:.1%})'
    
    mem_mb = df.memory_usage().sum() / 1024**2
    summary['Memory (est.)'] = f'{mem_mb:.1f} MB'
    
    return FirstContactReport(
        checks=pd.DataFrame(checks),
        summary=summary,
        dataset_name=dataset_name
    )


__all__ = [
    'first_contact_check',
    'FirstContactReport',
]
