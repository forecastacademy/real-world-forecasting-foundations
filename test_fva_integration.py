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
    # Note: segment_metrics needs abs_error and sum_actual columns, which are dropped after compute_metrics
    # We need to recompute those from the original data
    metric_level_with_segment = results.metric_level.copy()

    # Get segment mapping from original data
    segment_map = df.groupby('unique_id')['segment'].first()
    metric_level_with_segment['segment'] = metric_level_with_segment['unique_id'].map(segment_map)

    # Recreate abs_error and sum_actual for segment computation
    metric_level_detail = df.groupby(['unique_id', 'model']).agg(
        y_actual=('y', 'sum'),
        y_pred=('y_pred', 'sum'),
    ).reset_index()
    metric_level_detail['error'] = metric_level_detail['y_pred'] - metric_level_detail['y_actual']
    metric_level_detail['abs_error'] = metric_level_detail['error'].abs()
    metric_level_detail['sum_actual'] = metric_level_detail['y_actual']
    metric_level_detail = metric_level_detail.rename(columns={'unique_id': 'unique_id', 'model': 'model'})

    # Merge back to metric_level
    metric_level_with_segment = metric_level_with_segment.merge(
        metric_level_detail[['unique_id', 'model', 'abs_error', 'error', 'sum_actual']],
        on=['unique_id', 'model'],
        how='left'
    )

    # Also add beat columns for segment computation
    metric_level_detail['beat_sum'] = 0
    metric_level_detail['beat_count'] = 1
    metric_level_with_segment = metric_level_with_segment.merge(
        metric_level_detail[['unique_id', 'model', 'beat_sum', 'beat_count']],
        on=['unique_id', 'model'],
        how='left',
        suffixes=('', '_y')
    )

    # Create a temporary results object with the enhanced metric_level
    from src.metrics import MetricResults
    results_with_cols = MetricResults(metric_level=metric_level_with_segment, portfolio=results.portfolio)

    segments = calc.segment_metrics(results_with_cols, segment_cols=['segment'], metrics=['fva'])

    assert 'segment' in segments, "segment key not in results"
    assert 'fva' in segments['segment'].columns, "FVA not in segment results"

    print("✓ Segment FVA test passed")

if __name__ == "__main__":
    test_fva_absolute()
    test_fva_relative()
    test_fva_segment()
    print("\n✓ All FVA integration tests passed!")
