"""
Data Inspection Utilities
=========================

Functions for inspecting and diagnosing time series data quality.
"""

import pandas as pd


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


__all__ = [
    'check_gaps',
    'aggregate_calendar_to_weekly',
]
