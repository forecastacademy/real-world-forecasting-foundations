# FVA (Forecast Value Add) Metric Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FVA (Forecast Value Add) as a first-class metric option to MetricsCalculator, enabling model comparison via percentage-point improvement against anchor model's WMAPE.

**Architecture:** FVA is computed per-cutoff (like beat_rate) but then **recomputed from aggregated WMAPE sums** at every aggregation boundary (metric_level finalize, portfolio, segment). This ensures proper weighting across all levels. FVA is defined as `anchor_wmape - model_wmape` in absolute terms, with optional `fva_relative` flag for percentage improvement: `(anchor_wmape - model_wmape) / anchor_wmape * 100`. Division-by-zero in relative mode returns `np.nan`.

**Tech Stack:** pandas, numpy (existing); no new dependencies.

---

## Task 1: Update Constants and Initialize Parameter

**Files:**
- Modify: `src/metrics.py:52` (VALID_METRICS)
- Modify: `src/metrics.py:111-126` (__init__ method)

**Step 1: Add "fva" to VALID_METRICS**

In `src/metrics.py` line 52, change:
```python
VALID_METRICS = {"wmape", "bias", "jitter", "beat_rate"}
```

To:
```python
VALID_METRICS = {"wmape", "bias", "jitter", "beat_rate", "fva"}
```

**Step 2: Add fva_relative parameter to __init__**

In `src/metrics.py` line 111-126, change the `__init__` method signature and add the parameter:
```python
def __init__(
    self,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    id_col: str = DEFAULT_ID_COL,
    time_col: str = DEFAULT_TIME_COL,
    target_col: str = DEFAULT_TARGET_COL,
    cutoff_col: str = DEFAULT_CUTOFF_COL,
    model_col: str = DEFAULT_MODEL_COL,
    fva_relative: bool = False,
):
    """Initialize MetricsCalculator with configuration."""
    self.anchor_model = anchor_model
    self.id_col = id_col
    self.time_col = time_col
    self.target_col = target_col
    self.cutoff_col = cutoff_col
    self.model_col = model_col
    self.fva_relative = fva_relative
```

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: add fva_relative parameter to MetricsCalculator init"
```

---

## Task 2: Implement _compute_fva() Method

**Files:**
- Modify: `src/metrics.py` (add new method after `_compute_beat_rate()`)

**Step 1: Add _compute_fva() method**

After `_compute_beat_rate()` (around line 548), add:

```python
def _compute_fva(self, df: pd.DataFrame, metric_level: str) -> pd.DataFrame:
    """
    Compute FVA (Forecast Value Add) against anchor model.

    FVA = anchor_model_wmape - model_wmape (in percentage points by default)
    When fva_relative=True: FVA = (anchor_wmape - model_wmape) / anchor_wmape × 100

    Comparison done within each cutoff. At finalize/portfolio/segment stages,
    FVA is recomputed from aggregated WMAPE values to ensure proper weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated data with wmape column.
    metric_level : str
        The metric_level column name (e.g., "item_id").

    Returns
    -------
    pd.DataFrame
        Input with added fva column.
    """
    # Get anchor WMAPE per (metric_level, cutoff)
    merge_keys = [metric_level, self.cutoff_col]

    anchor_wmape = (
        df[df[self.model_col] == self.anchor_model]
        [merge_keys + ["wmape"]]
        .rename(columns={"wmape": "anchor_wmape"})
    )

    # Merge and compute FVA
    df = df.merge(anchor_wmape, on=merge_keys, how="left")

    if self.fva_relative:
        df = df.assign(
            fva=lambda x: np.where(
                x["anchor_wmape"] != 0,
                ((x["anchor_wmape"] - x["wmape"]) / x["anchor_wmape"]) * 100,
                np.nan,
            )
        )
    else:
        df = df.assign(fva=lambda x: x["anchor_wmape"] - x["wmape"])

    df = df.drop(columns=["anchor_wmape"])

    return df
```

**Step 2: Verify syntax (no test yet, just import)**

```bash
python -c "from src.metrics import MetricsCalculator; print('✓ Syntax OK')"
```

Expected: `✓ Syntax OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: implement _compute_fva() method for per-cutoff FVA calculation"
```

---

## Task 3: Integrate FVA into compute_metrics() Step 3

**Files:**
- Modify: `src/metrics.py:205-220` (compute_metrics Step 3)

**Step 1: Add FVA computation to Step 3**

In `src/metrics.py` around line 215-219, after the beat_rate block, add:

```python
if "fva" in metrics:
    metric_level_df = self._compute_fva(metric_level_df, metric_level)
```

The full block should look like:
```python
if "beat_rate" in metrics:
    metric_level_df = self._compute_beat_rate(metric_level_df, metric_level)
    drop_cols.extend(['beat_count','beat_sum'])
if "fva" in metrics:
    metric_level_df = self._compute_fva(metric_level_df, metric_level)
if "jitter" in metrics:
    metric_level_df = self._compute_jitter(metric_level_df, metric_level)
```

**Step 2: Verify syntax**

```bash
python -c "from src.metrics import MetricsCalculator; print('✓ Integration OK')"
```

Expected: `✓ Integration OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: integrate FVA computation into compute_metrics() Step 3"
```

---

## Task 4: Recompute FVA in _finalize_metric_level()

**Files:**
- Modify: `src/metrics.py:445-451` (_finalize_metric_level)

**Step 1: Add FVA recomputation after WMAPE/bias recomputation**

In `src/metrics.py` around line 450 (after the bias block, before return), add:

```python
# Recompute FVA from aggregated WMAPE values (proper weighting across cutoffs)
if "fva" in metric_level_df.columns:
    # Get anchor WMAPE per metric_level (after aggregation)
    anchor_df = metric_level_df[
        metric_level_df[self.model_col] == self.anchor_model
    ].copy()
    anchor_wmape = anchor_df[[metric_level, "wmape"]].rename(
        columns={"wmape": "anchor_wmape"}
    )

    # Merge anchor WMAPE back to all models
    metric_level_df = metric_level_df.merge(anchor_wmape, on=metric_level, how="left")

    # Compute FVA with divide-by-zero protection
    if self.fva_relative:
        metric_level_df = metric_level_df.assign(
            fva=lambda x: np.where(
                x["anchor_wmape"] != 0,
                ((x["anchor_wmape"] - x["wmape"]) / x["anchor_wmape"]) * 100,
                np.nan,
            )
        )
    else:
        metric_level_df = metric_level_df.assign(
            fva=lambda x: x["anchor_wmape"] - x["wmape"]
        )

    metric_level_df = metric_level_df.drop(columns=["anchor_wmape"])
```

**Step 2: Verify syntax**

```bash
python -c "from src.metrics import MetricsCalculator; print('✓ Finalize OK')"
```

Expected: `✓ Finalize OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: recompute FVA from aggregated WMAPE in _finalize_metric_level()"
```

---

## Task 5: Add FVA to _aggregate_to_portfolio()

**Files:**
- Modify: `src/metrics.py:629-637` (_aggregate_to_portfolio)

**Step 1: Add FVA computation block after beat_rate**

In `src/metrics.py` after line 635 (after the beat_rate block), add:

```python
if "fva" in metrics:
    # Recompute FVA from aggregated WMAPE at portfolio level
    wmape_agg = metric_level_df.groupby(self.model_col).agg(
        _total_abs_error=("abs_error", "sum"),
        _total_actual=("sum_actual", "sum"),
    )
    wmape_agg["wmape"] = wmape_agg["_total_abs_error"] / wmape_agg["_total_actual"]

    anchor_wmape = wmape_agg.loc[self.anchor_model, "wmape"]

    if self.fva_relative:
        portfolio["fva"] = np.where(
            anchor_wmape != 0,
            ((anchor_wmape - wmape_agg["wmape"]) / anchor_wmape) * 100,
            np.nan,
        )
    else:
        portfolio["fva"] = anchor_wmape - wmape_agg["wmape"]
```

**Step 2: Verify syntax**

```bash
python -c "from src.metrics import MetricsCalculator; print('✓ Portfolio OK')"
```

Expected: `✓ Portfolio OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: add FVA recomputation to _aggregate_to_portfolio()"
```

---

## Task 6: Add FVA to segment_metrics()

**Files:**
- Modify: `src/metrics.py:709-722` (segment_metrics beat_rate block)

**Step 1: Add FVA computation after beat_rate block**

In `src/metrics.py` after the beat_rate block (around line 722), add:

```python
if "fva" in metrics:
    # Recompute FVA from aggregated WMAPE at segment level
    segment_wmape = df.groupby([segment_col, self.model_col]).agg(
        _total_abs_error=("abs_error", "sum"),
        _total_actual=("sum_actual", "sum"),
    ).reset_index()
    segment_wmape["wmape"] = (
        segment_wmape["_total_abs_error"] / segment_wmape["_total_actual"]
    )

    # Get anchor WMAPE per segment
    anchor_by_segment = segment_wmape[
        segment_wmape[self.model_col] == self.anchor_model
    ].copy()
    anchor_by_segment = anchor_by_segment[[segment_col, "wmape"]].rename(
        columns={"wmape": "anchor_wmape"}
    )

    segment_wmape = segment_wmape.merge(anchor_by_segment, on=segment_col, how="left")

    if self.fva_relative:
        segment_wmape["fva"] = np.where(
            segment_wmape["anchor_wmape"] != 0,
            (
                (segment_wmape["anchor_wmape"] - segment_wmape["wmape"])
                / segment_wmape["anchor_wmape"]
            )
            * 100,
            np.nan,
        )
    else:
        segment_wmape["fva"] = segment_wmape["anchor_wmape"] - segment_wmape["wmape"]

    segment_df = segment_df.merge(
        segment_wmape[[segment_col, self.model_col, "fva"]],
        on=[segment_col, self.model_col],
    )
```

**Step 2: Verify syntax**

```bash
python -c "from src.metrics import MetricsCalculator; print('✓ Segment OK')"
```

Expected: `✓ Segment OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: add FVA recomputation to segment_metrics()"
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Create: `test_fva_integration.py` (temporary test file)

**Step 1: Write comprehensive FVA test**

Create a test file with:

```python
import pandas as pd
import numpy as np
from src.metrics import MetricsCalculator

def test_fva_absolute():
    """Test absolute FVA computation"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='W')
    cutoffs = pd.date_range('2024-02-01', periods=4, freq='4W')

    data = []
    for cutoff in cutoffs:
        for unique_id in ['SKU_1', 'SKU_2', 'SKU_3']:
            for date in dates[dates > cutoff][:10]:
                data.append({
                    'unique_id': unique_id,
                    'ds': date,
                    'y': np.random.randint(50, 150),
                    'y_pred': np.random.randint(50, 150),
                    'model': 'SN52' if unique_id == 'SKU_1' else 'Model_A',
                    'cutoff': cutoff
                })

    df = pd.DataFrame(data)
    calc = MetricsCalculator(anchor_model='SN52', fva_relative=False)

    results = calc.compute_metrics(
        df,
        error_level='unique_id',
        metric_level='unique_id',
        timesteps=None,
        metrics=['wmape', 'fva']
    )

    # Verify FVA column exists
    assert 'fva' in results.metric_level.columns, "FVA not in metric_level"
    assert 'fva' in results.portfolio.columns, "FVA not in portfolio"

    # Verify anchor model has FVA = 0 (by definition, can't beat itself)
    anchor_fva = results.portfolio[results.portfolio['model'] == 'SN52']['fva'].values
    assert len(anchor_fva) > 0 and abs(anchor_fva[0]) < 1e-10, \
        f"Anchor FVA should be ~0, got {anchor_fva[0]}"

    print("✓ Absolute FVA test passed")

def test_fva_relative():
    """Test relative FVA computation"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='W')
    cutoffs = pd.date_range('2024-02-01', periods=4, freq='4W')

    data = []
    for cutoff in cutoffs:
        for unique_id in ['SKU_1', 'SKU_2', 'SKU_3']:
            for date in dates[dates > cutoff][:10]:
                data.append({
                    'unique_id': unique_id,
                    'ds': date,
                    'y': np.random.randint(50, 150),
                    'y_pred': np.random.randint(50, 150),
                    'model': 'SN52' if unique_id == 'SKU_1' else 'Model_A',
                    'cutoff': cutoff
                })

    df = pd.DataFrame(data)
    calc = MetricsCalculator(anchor_model='SN52', fva_relative=True)

    results = calc.compute_metrics(
        df,
        error_level='unique_id',
        metric_level='unique_id',
        timesteps=None,
        metrics=['wmape', 'fva']
    )

    # Verify FVA column exists
    assert 'fva' in results.metric_level.columns, "FVA not in metric_level"
    assert 'fva' in results.portfolio.columns, "FVA not in portfolio"

    # Relative FVA should be 0 for anchor (within floating point)
    anchor_fva = results.portfolio[results.portfolio['model'] == 'SN52']['fva'].values
    assert len(anchor_fva) > 0 and abs(anchor_fva[0]) < 1e-10, \
        f"Anchor relative FVA should be ~0, got {anchor_fva[0]}"

    print("✓ Relative FVA test passed")

def test_fva_segment():
    """Test FVA in segment_metrics"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='W')
    cutoffs = pd.date_range('2024-02-01', periods=4, freq='4W')

    data = []
    segment_id = 0
    for cutoff in cutoffs:
        for unique_id in ['SKU_1', 'SKU_2', 'SKU_3']:
            for date in dates[dates > cutoff][:10]:
                data.append({
                    'unique_id': unique_id,
                    'ds': date,
                    'y': np.random.randint(50, 150),
                    'y_pred': np.random.randint(50, 150),
                    'model': 'SN52' if unique_id == 'SKU_1' else 'Model_A',
                    'cutoff': cutoff,
                    'segment': 'A' if segment_id % 2 == 0 else 'B',
                })
                segment_id += 1

    df = pd.DataFrame(data)
    calc = MetricsCalculator(anchor_model='SN52', fva_relative=False)

    results = calc.compute_metrics(
        df,
        error_level='unique_id',
        metric_level='unique_id',
        timesteps=None,
        metrics=['wmape', 'fva']
    )

    # Add segment column to metric_level for segment_metrics
    results.metric_level['segment'] = df.groupby('unique_id')['segment'].first().values

    segments = calc.segment_metrics(results, segment_cols=['segment'], metrics=['fva'])

    assert 'segment' in segments, "segment key not in results"
    assert 'fva' in segments['segment'].columns, "FVA not in segment results"

    print("✓ Segment FVA test passed")

if __name__ == "__main__":
    test_fva_absolute()
    test_fva_relative()
    test_fva_segment()
    print("\n✓ All FVA integration tests passed!")
```

**Step 2: Run the test**

```bash
python test_fva_integration.py
```

Expected output:
```
✓ Absolute FVA test passed
✓ Relative FVA test passed
✓ Segment FVA test passed

✓ All FVA integration tests passed!
```

**Step 3: Commit**

```bash
git add test_fva_integration.py
git commit -m "test: add comprehensive FVA integration tests"
```

**Step 4: Clean up test file**

```bash
rm test_fva_integration.py
git add -A
git commit -m "test: remove temporary FVA integration test file"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `src/metrics.py:1-33` (module docstring)
- Modify: `src/metrics.py:52` (VALID_METRICS reference)
- Modify: `src/metrics.py:60-76` (MetricResults docstring)

**Step 1: Update module docstring**

In `src/metrics.py` line 15, change:
```python
- Four core metrics: WMAPE, BIAS, JITTER, BEAT_RATE
```

To:
```python
- Five core metrics: WMAPE, BIAS, JITTER, BEAT_RATE, FVA
```

And in the example (line 28), change:
```python
...     metrics=["wmape", "bias", "jitter", "beat_rate"]
```

To:
```python
...     metrics=["wmape", "bias", "jitter", "beat_rate", "fva"]
```

**Step 2: Update MetricResults docstring**

In `src/metrics.py` lines 68-73, add to the columns list:
```python
        Columns include: metric_level, model, sum_forecast, sum_actual, error,
        abs_error, wmape, bias, beat_rate, beat_sum, beat_count, fva, jitter
```

And in the portfolio docstring (line 73), add:
```python
        Columns include: model, wmape, bias, jitter, beat_rate, fva
```

**Step 3: Verify docstring updates**

```bash
python -c "from src.metrics import MetricsCalculator; help(MetricsCalculator)" | head -30
```

Expected: Should see FVA mentioned in docstring

**Step 4: Commit**

```bash
git add src/metrics.py
git commit -m "docs: update docstrings to include FVA metric"
```

---

## Summary

**Total Steps:** 8 tasks, ~15 commits

**Key Implementation Details:**
1. FVA is added as first-class metric alongside wmape, bias, jitter, beat_rate
2. Per-cutoff: FVA computed as `anchor_wmape - model_wmape`
3. After aggregation: FVA recomputed from aggregated WMAPE sums at every level
4. Division-by-zero in relative mode returns `np.nan`
5. Three aggregation boundaries handle FVA: finalize, portfolio, segment

**Testing Coverage:**
- Absolute FVA mode (default)
- Relative FVA mode (with fva_relative=True)
- Segment-level FVA computation
- Anchor model FVA validation (should be ~0)

**Commits Follow Pattern:**
- Feature commits (constants, methods, integrations)
- Test commits (integration validation)
- Documentation commits (docstring updates)

