"""
Forecast Evaluation Metrics
============================

Compute executive-level forecast evaluation metrics at business decision grain.

This module provides the MetricsCalculator class for computing comprehensive
forecast performance metrics, including flexible aggregation, anchor model
comparisons, and segment analysis.

Features
--------
- Flexible error and metric grain aggregation
- Time-based filtering (horizons, timesteps)
- Four core metrics: WMAPE, BIAS, JITTER, BEAT_RATE
- Dual-grain output: detail metrics + portfolio summary
- Segment analysis for slicing by business dimensions

Usage
-----
    >>> from src import MetricsCalculator, MetricResults
    >>> calc = MetricsCalculator(anchor_model="SN52")
    >>> results = calc.compute_metrics(
    ...     df,
    ...     error_level="unique_id",
    ...     metric_level="item_id",
    ...     timesteps=[(5, 13)],
    ...     metrics=["wmape", "bias", "jitter", "beat_rate"]
    ... )
    >>> results.metric_level  # item_id × model × cutoff
    >>> results.portfolio     # model level
    >>> segments = calc.segment_metrics(results, segment_cols=["dept_id"])
"""

from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel

# ============================================================================
# Constants
# ============================================================================

DEFAULT_ANCHOR_MODEL = "SN52"
DEFAULT_ID_COL = "unique_id"
DEFAULT_TIME_COL = "ds"
DEFAULT_TARGET_COL = "y"
DEFAULT_CUTOFF_COL = "cutoff"
DEFAULT_MODEL_COL = "model"

VALID_METRICS = {"wmape", "bias", "jitter", "beat_rate", "fva"}


# ============================================================================
# Pydantic Models
# ============================================================================


class MetricResults(BaseModel):
    """Container for metric computation results at two grains.

    Attributes
    ----------
    metric_level : pd.DataFrame
        Detail metrics at metric_level × model grain (cutoff removed to ensure
        single summary per groupbykeys). Jitter computed across all cutoffs;
        beat_rate averaged across cutoffs (0-1 scale).
        Columns include: metric_level, model, sum_forecast, sum_actual, error,
        abs_error, wmape, bias, beat_rate, jitter
    portfolio : pd.DataFrame
        Portfolio summary aggregated to model level.
        Columns include: model, wmape, bias, jitter, beat_rate
    """
    metric_level: pd.DataFrame
    portfolio: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# MetricsCalculator Class
# ============================================================================


class MetricsCalculator:
    """
    Compute forecast evaluation metrics at flexible business decision grains.

    Orchestrates the complete pipeline: filtering, aggregation, metric
    computation, and portfolio summarization. Supports flexible grain
    specification for both error location and metric aggregation.

    Parameters
    ----------
    anchor_model : str, default="SN52"
        Baseline model for beat rate calculations.
    id_col : str, default="unique_id"
        Time series identifier column (SKU × Store combinations).
    time_col : str, default="ds"
        Timestamp column name.
    target_col : str, default="y"
        Actual values column name.
    cutoff_col : str, default="cutoff"
        Forecast cutoff date column name.
    model_col : str, default="model"
        Model identifier column name.
    """

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

    # ========================================================================
    # Main Orchestration
    # ========================================================================

    def compute_metrics(
        self,
        df: pd.DataFrame,
        error_level: str,
        metric_level: str,
        timesteps: Optional[List[Tuple[int, int]]] = None,
        metrics: List[str] = ["wmape", "bias", "jitter", "beat_rate"],
    ) -> MetricResults:
        """
        Compute metrics at flexible grains with time filtering.

        Orchestrates the complete pipeline:
        1. Filter by timesteps (if provided)
        2. Aggregate from error_level to metric_level (sum forecasts/actuals)
        3. Compute errors at metric_level grain
        4. Compute requested metrics (jitter always across all cutoffs)
        5. Aggregate to portfolio (model) level

        Parameters
        ----------
        df : pd.DataFrame
            Backtest data with columns: model, error_level, time_col,
            target_col, cutoff_col, y_pred (forecast), and y (actual).
        error_level : str
            Grain at which forecast errors exist in raw data
            (e.g., "unique_id" for store-level data).
        metric_level : str
            Grain to aggregate to for metrics computation
            (e.g., "item_id" for SKU-level metrics).
        timesteps : list[tuple[int, int]], optional
            Time ranges to include. Each tuple is (start_week, end_week).
            If None, uses all data. Example: [(1, 4), (5, 13)] for
            short-term and mid-term horizons.
        metrics : list[str], default=["wmape", "bias", "jitter", "beat_rate"]
            Metrics to compute. Must be subset of
            {"wmape", "bias", "jitter", "beat_rate"}.

        Returns
        -------
        MetricResults
            Container with:
            - metric_level: DataFrame at metric_level × model grain (cutoff removed from
              final output; jitter computed across all cutoffs)
            - portfolio: DataFrame at model level with aggregated metrics

        Raises
        ------
        ValueError
            If jitter or beat_rate requested but cutoff_col not found in dataframe.
        """
        # Validate inputs
        invalid = set(metrics) - VALID_METRICS
        if invalid:
            raise ValueError(
                f"Invalid metrics: {invalid}. "
                f"Must be subset of {VALID_METRICS}"
            )

        # Validate cutoff column exists if jitter or beat_rate requested
        if ("jitter" in metrics or "beat_rate" in metrics) and self.cutoff_col not in df.columns:
            raise ValueError(
                f"Metrics {[m for m in ['jitter', 'beat_rate'] if m in metrics]} "
                f"require cutoff column '{self.cutoff_col}' which not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        # Step 1: Filter by timesteps
        df_filtered = self._filter_by_timesteps(df, timesteps)

        # Step 2: Aggregate to metric_level and compute errors
        agg_df = self._aggregate_to_metric_level(df_filtered, metric_level)
        agg_df = self._compute_aggregated_errors(agg_df)

        # Step 3: Compute requested metrics
        metric_level_df = agg_df.copy()
        
        drop_cols = []
        if "wmape" in metrics:
            metric_level_df = self._compute_wmape(metric_level_df)
            drop_cols.extend(['sum_forecast','sum_actual','abs_error','error'])
        if "bias" in metrics:
            metric_level_df = self._compute_bias(metric_level_df)

        if "beat_rate" in metrics:
            metric_level_df = self._compute_beat_rate(metric_level_df, metric_level)
            drop_cols.extend(['beat_count','beat_sum'])
        if "jitter" in metrics:
            metric_level_df = self._compute_jitter(metric_level_df, metric_level)

        # Step 4: Remove cutoff from final metric_level_df to ensure single summary per groupbykeys
        # (metric_level, model are the groupby keys; cutoff is only used for internal computation)
        metric_level_df = self._finalize_metric_level(metric_level_df, metric_level)

        # Step 5: Aggregate to portfolio (model) level
        portfolio_df = self._aggregate_to_portfolio(metric_level_df, metrics)

        metric_level_df = metric_level_df.drop(drop_cols,axis=1)

        return MetricResults(metric_level=metric_level_df, portfolio=portfolio_df)

    def compute_metrics_batch(
        self,
        df: pd.DataFrame,
        metric_levels: Dict[str, str],
        timesteps: Optional[List[Tuple[int, int]]] = None,
        metrics: List[str] = ["wmape", "bias", "jitter", "beat_rate"],
    ) -> Dict[str, MetricResults]:
        """
        Compute metrics for multiple metric_level grains in one call.

        Takes a dictionary of error_level:metric_level pairs and returns
        a dictionary of MetricResults, one per grain. Cutoff removed from
        final output; jitter computed across all cutoffs.

        Parameters
        ----------
        df : pd.DataFrame
            Backtest data with columns: model, error_level, time_col,
            target_col, cutoff_col, y_pred (forecast), and y (actual).
        metric_levels : dict[str, str]
            Dictionary mapping error_level to metric_level columns.
            Example: {"unique_id": "item_id", "store_id": "region_id"}
        timesteps : list[tuple[int, int]], optional
            Time ranges to include. Each tuple is (start_week, end_week).
        metrics : list[str], default=["wmape", "bias", "jitter", "beat_rate"]
            Metrics to compute for all grains.

        Returns
        -------
        dict[str, MetricResults]
            Dictionary keyed by metric_level column name, with MetricResults
            as values. Example: {"item_id": MetricResults(...), "region_id": ...}

        Raises
        ------
        ValueError
            If metric_levels is empty or jitter requested without cutoff column.
        """
        if not metric_levels:
            raise ValueError("metric_levels dictionary cannot be empty")

        results = {}

        for error_level, metric_level in metric_levels.items():
            results[metric_level] = self.compute_metrics(
                df=df,
                error_level=error_level,
                metric_level=metric_level,
                timesteps=timesteps,
                metrics=metrics,
            )

        return results

    # ========================================================================
    # Data Preparation & Filtering
    # ========================================================================

    def _filter_by_timesteps(
        self,
        df: pd.DataFrame,
        timesteps: Optional[List[Tuple[int, int]]],
    ) -> pd.DataFrame:
        """
        Filter data to specified timestep ranges.

        Parameters
        ----------
        df : pd.DataFrame
            Input backtest data.
        timesteps : list[tuple[int, int]] or None
            Time ranges in weeks: [(start_week, end_week), ...].
            If None, returns full dataframe.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        if timesteps is None:
            return df.copy()

        df = df.copy()

        # Compute weeks from cutoff
        df["_weeks_from_cutoff"] = (
            (df[self.time_col] - df[self.cutoff_col]).dt.days / 7
        )

        # Filter to specified ranges
        masks = []
        for start_week, end_week in timesteps:
            mask = (
                (df["_weeks_from_cutoff"] >= start_week)
                & (df["_weeks_from_cutoff"] <= end_week)
            )
            masks.append(mask)

        df_filtered = df[pd.concat(masks, axis=1).any(axis=1)].copy()
        df_filtered = df_filtered.drop(columns=["_weeks_from_cutoff"])

        return df_filtered

    def _aggregate_to_metric_level(
        self,
        df: pd.DataFrame,
        metric_level: str,
    ) -> pd.DataFrame:
        """
        Aggregate from error_level to metric_level.

        Groups by (metric_level, model, cutoff). Sums forecasts/actuals.
        Applies hygiene filter to exclude windows with zero actual demand.
        Cutoff is kept for internal metric computation and removed in final output.

        Parameters
        ----------
        df : pd.DataFrame
            Filtered backtest data.
        metric_level : str
            Column to aggregate to (e.g., "item_id").

        Returns
        -------
        pd.DataFrame
            Aggregated data with columns: metric_level, model, cutoff,
            sum_forecast, sum_actual.
        """
        group_cols = [metric_level, self.model_col, self.cutoff_col]

        agg_df = (
            df.groupby(group_cols, as_index=False)
            .agg(sum_forecast=("y_pred", "sum"), sum_actual=(self.target_col, "sum"))
            .copy()
        )

        return agg_df

    def _compute_aggregated_errors(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute error metrics at aggregation grain.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated data with sum_forecast and sum_actual.

        Returns
        -------
        pd.DataFrame
            Input with added error columns: error, abs_error.
        """
        agg_df = agg_df.assign(
            error=lambda x: x["sum_forecast"] - x["sum_actual"],
        ).assign(
            abs_error=lambda x: x["error"].abs(),
        )

        return agg_df

    def _finalize_metric_level(
        self,
        metric_level_df: pd.DataFrame,
        metric_level: str,
    ) -> pd.DataFrame:
        """
        Finalize metric_level dataframe by removing cutoff column.

        Ensures a single summary row per (metric_level, model) groupby keys.
        Cutoff is used internally for metric computation but not included in final output.
        Sums base columns across cutoffs and recomputes wmape/bias; averages beat_rate.

        Parameters
        ----------
        metric_level_df : pd.DataFrame
            Computed metrics at (metric_level, model, cutoff) grain.
        metric_level : str
            The metric_level column name (e.g., "item_id").

        Returns
        -------
        pd.DataFrame
            Metric dataframe with cutoff removed, aggregated to
            (metric_level, model) grain. wmape/bias recomputed from sums;
            beat_rate averaged across cutoffs.
        """
        # Define groupby keys (exclude cutoff)
        groupby_keys = [metric_level, self.model_col]

        # Sum base columns across cutoffs; jitter stays same (already cross-cutoff)
        # Use named aggregation syntax to rename columns during aggregation
        metric_level_df = (
            metric_level_df.groupby(groupby_keys, as_index=False)
            .agg(
                sum_forecast=("sum_forecast", "sum"),
                sum_actual=("sum_actual", "sum"),
                error=("error", "sum"),
                abs_error=("abs_error", "sum"),
                beat_sum=("beat_indicator", "sum"),      # Sum indicators, rename to beat_sum
                beat_count=("beat_count", "sum"),        # Sum counts for weighting
                jitter=("jitter", "first"),              # Already computed as std across cutoffs
            )
            .copy()
        )

        # Recompute wmape and bias from aggregated base columns
        if "wmape" in metric_level_df.columns or "abs_error" in metric_level_df.columns:
            metric_level_df = metric_level_df.assign(
                wmape=lambda x: x["abs_error"] / x["sum_actual"]
            )

        if "bias" in metric_level_df.columns or "error" in metric_level_df.columns:
            metric_level_df = metric_level_df.assign(
                bias=lambda x: x["error"] / x["sum_actual"]
            )

        # Compute beat_rate from aggregated sums (proper weighting across cutoffs)
        if "beat_sum" in metric_level_df.columns:
            metric_level_df = metric_level_df.assign(
                beat_rate=lambda x: x["beat_sum"] / x["beat_count"]
            )

        return metric_level_df

    # ========================================================================
    # Metrics Computation
    # ========================================================================

    def _compute_wmape(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute WMAPE (Weighted Mean Absolute Percentage Error).

        WMAPE = abs_error / sum_actual at metric_level grain.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated data with error columns.

        Returns
        -------
        pd.DataFrame
            Input with added wmape column.
        """
        df = df.assign(
            wmape=lambda x: x["abs_error"] / x["sum_actual"]
        )
        return df

    def _compute_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BIAS (directional forecast bias as %).

        Bias = error / sum_actual at metric_level grain.
        Positive = overprediction, Negative = underprediction.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated data with error columns.

        Returns
        -------
        pd.DataFrame
            Input with added bias column.
        """
        df = df.assign(
            bias=lambda x: x["error"] / x["sum_actual"]
        )
        return df

    def _compute_beat_rate(
        self, df: pd.DataFrame, metric_level: str
    ) -> pd.DataFrame:
        """
        Compute beat rate against anchor model.

        A model "beats" the anchor if its WMAPE is lower.
        Comparison done within each cutoff.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated data with wmape column.
        metric_level : str
            The metric_level column name (e.g., "item_id").

        Returns
        -------
        pd.DataFrame
            Input with added beat_indicator column (0/1).
        """
        # Get anchor performance (compare within each cutoff)
        merge_keys = [metric_level, self.cutoff_col]

        anchor_wmape = (
            df[df[self.model_col] == self.anchor_model]
            [merge_keys + ["wmape"]]
            .rename(columns={"wmape": "anchor_wmape"})
        )

        # Merge and compute beat indicator
        df = df.merge(
            anchor_wmape,
            on=merge_keys,
            how="left",
        )

        df = df.assign(
            beat_indicator=lambda x: (x["wmape"] < x["anchor_wmape"]).astype(int)
        )

        df = df.assign(
            beat_count=1  # Count observations for proper aggregation across cutoffs
        )

        df = df.drop(columns=["anchor_wmape"])

        return df

    def _compute_jitter(
        self, df: pd.DataFrame, metric_level: str
    ) -> pd.DataFrame:
        """
        Compute JITTER (stability of WMAPE across cutoffs).

        For each (model, metric_level_item): std(wmape across all cutoffs).

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated data with wmape column.
        metric_level : str
            The metric_level column name (e.g., "item_id").

        Returns
        -------
        pd.DataFrame
            Input with added jitter column (std of wmape across all cutoffs per series).
        """
        # Always compute jitter per series (std across all cutoffs)
        series_jitter = (
            df.groupby([self.model_col, metric_level])["wmape"]
            .std()
            .reset_index()
            .rename(columns={"wmape": "jitter"})
        )

        # Merge back to detail
        df = df.merge(
            series_jitter,
            on=[self.model_col, metric_level],
            how="left",
        )

        return df

    # ========================================================================
    # Portfolio Aggregation
    # ========================================================================

    def _aggregate_to_portfolio(
        self, metric_level_df: pd.DataFrame, metrics: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate metric_level results to portfolio (model) level.

        Parameters
        ----------
        metric_level_df : pd.DataFrame
            Metric-level results.
        metrics : list[str]
            Requested metrics.

        Returns
        -------
        pd.DataFrame
            Portfolio metrics aggregated by model.
        """
        # Initialize portfolio with model column
        portfolio = metric_level_df.groupby(self.model_col).first()[[]]

        if "wmape" in metrics:
            # Recompute at portfolio: sum(all abs_error) / sum(all sum_actual)
            wmape_agg = metric_level_df.groupby(self.model_col).agg(
                wmape=("abs_error", "sum"),
                _total_actual=("sum_actual", "sum"),
            )
            portfolio["wmape"] = wmape_agg["wmape"] / wmape_agg["_total_actual"]

        if "bias" in metrics:
            # Recompute at portfolio: sum(all error) / sum(all sum_actual)
            bias_agg = metric_level_df.groupby(self.model_col).agg(
                bias=("error", "sum"),
                _total_actual=("sum_actual", "sum"),
            )
            portfolio["bias"] = bias_agg["bias"] / bias_agg["_total_actual"]

        if "jitter" in metrics:
            # Mean of series-level jitter values
            jitter_agg = metric_level_df.groupby(self.model_col)["jitter"].mean()
            portfolio["jitter"] = jitter_agg

        if "beat_rate" in metrics:
            # Recompute beat_rate from raw sums to maintain proper weighting
            beat_agg = metric_level_df.groupby(self.model_col).agg(
                beat_sum=("beat_sum", "sum"),
                beat_count=("beat_count", "sum"),
            )
            portfolio["beat_rate"] = (beat_agg["beat_sum"] / beat_agg["beat_count"]) * 100

        return portfolio.reset_index()

    # ========================================================================
    # Segment Analysis
    # ========================================================================

    def segment_metrics(
        self,
        metric_results: MetricResults,
        segment_cols: List[str],
        metrics: List[str] = ["wmape", "bias", "beat_rate"],
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute metrics sliced by business segments.

        Takes metric_level results and aggregates by segment column(s).
        Single responsibility: segment-level aggregation only.

        Parameters
        ----------
        metric_results : MetricResults
            Output from compute_metrics().
        segment_cols : list[str]
            Columns to segment by (e.g., ["dept_id", "abc_class"]).
            These columns must exist in metric_results.metric_level.
        metrics : list[str], default=["wmape", "bias", "beat_rate"]
            Metrics to compute per segment. Note: jitter not supported
            at segment grain (requires cross-cutoff variation per segment).

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary keyed by segment column name. Each value is a
            DataFrame with metrics aggregated by that segment.
            Columns: segment_col, model, wmape, bias, beat_rate (as applicable).

        Raises
        ------
        ValueError
            If segment_cols not in metric_level.
        """
        df = metric_results.metric_level

        # Validate segment columns exist
        missing = set(segment_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Segment columns not found: {missing}. "
                f"Available: {list(df.columns)}"
            )

        segment_results = {}

        for segment_col in segment_cols:
            segment_df = (
                df.groupby([segment_col, self.model_col], as_index=False)
                .agg(
                    _total_abs_error=("abs_error", "sum"),
                    _total_error=("error", "sum"),
                    _total_actual=("sum_actual", "sum"),
                )
            )

            if "wmape" in metrics:
                segment_df["wmape"] = (
                    segment_df["_total_abs_error"] / segment_df["_total_actual"]
                )

            if "bias" in metrics:
                segment_df["bias"] = (
                    segment_df["_total_error"] / segment_df["_total_actual"]
                )

            if "beat_rate" in metrics:
                # Recompute beat_rate from raw sums at segment level
                beat_per_segment = df.groupby([segment_col, self.model_col]).agg(
                    beat_sum=("beat_sum", "sum"),
                    beat_count=("beat_count", "sum"),
                )
                beat_per_segment["beat_rate"] = (
                    beat_per_segment["beat_sum"] / beat_per_segment["beat_count"]
                ) * 100

                segment_df = segment_df.merge(
                    beat_per_segment[["beat_rate"]].reset_index(),
                    on=[segment_col, self.model_col],
                )

            # Clean up temp columns
            temp_cols = [c for c in segment_df.columns if c.startswith("_")]
            segment_df = segment_df.drop(columns=temp_cols)

            segment_results[segment_col] = segment_df

        return segment_results

    # ========================================================================
    # Visualization
    # ========================================================================

    def boxplot_against_anchor(
        self,
        df: pd.DataFrame,
        metric: str = "wmape",
        clip_quantiles: Optional[Tuple[float, float]] = (0.05, 0.95),
        show_points: bool = True,
        show_stats: bool = True,
        figsize: Tuple[int, int] = (900, 600),
        title: Optional[str] = None,
        color_scheme: str = "default",
    ) -> go.Figure:
        """
        Create boxplot comparing models against anchor.

        Parameters
        ----------
        df : pd.DataFrame
            Score DataFrame with model and metric columns.
        metric : str, default="wmape"
            Metric column to plot (e.g., "wmape", "bias").
        clip_quantiles : tuple[float, float] or None, default=(0.05, 0.95)
            Quantiles for outlier clipping. None disables clipping.
        show_points : bool, default=True
            Whether to overlay individual data points.
        show_stats : bool, default=True
            Whether to show summary statistics.
        figsize : tuple[int, int], default=(900, 600)
            Figure size as (width, height).
        title : str, optional
            Custom title. Auto-generated if None.
        color_scheme : str, default="default"
            Color palette: "default", "viridis", or "red_blue".

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive boxplot figure.
        """
        df = df.copy()

        # Create anchor flag
        df["is_anchor"] = df[self.model_col] == self.anchor_model

        # Clip outliers for visualization
        if clip_quantiles:
            lower = df[metric].quantile(clip_quantiles[0])
            upper = df[metric].quantile(clip_quantiles[1])
            df[f"{metric}_clipped"] = df[metric].clip(lower=lower, upper=upper)
            plot_metric = f"{metric}_clipped"
        else:
            plot_metric = metric

        # Calculate statistics for sorting and annotations
        model_stats = (
            df.groupby(self.model_col)[plot_metric]
            .agg(["median", "mean", "count"])
            .reset_index()
        )
        anchor_median = model_stats[
            model_stats[self.model_col] == self.anchor_model
        ]["median"].values[0]
        model_stats["delta_from_anchor"] = model_stats["median"] - anchor_median
        model_stats["delta_pct"] = (
            model_stats["delta_from_anchor"] / anchor_median * 100
        )

        # Sort models by median performance (anchor first)
        model_stats = model_stats.sort_values(
            [self.model_col], key=lambda x: x != self.anchor_model
        ).sort_values("median")
        model_order = model_stats[self.model_col].tolist()
        df[self.model_col] = pd.Categorical(
            df[self.model_col], categories=model_order, ordered=True
        )
        df = df.sort_values(self.model_col)

        # Color schemes
        color_schemes = {
            "default": {False: "#4C78A8", True: "#E45756"},
            "viridis": {False: "#440154", True: "#FDE724"},
            "red_blue": {False: "#3182bd", True: "#e6550d"},
        }
        colors = color_schemes.get(color_scheme, color_schemes["default"])

        # Create figure
        fig = go.Figure()

        # Add box plots
        for is_anchor in [False, True]:
            mask = df["is_anchor"] == is_anchor
            if not mask.any():
                continue

            fig.add_trace(
                go.Box(
                    y=df.loc[mask, self.model_col],
                    x=df.loc[mask, plot_metric],
                    name="Anchor" if is_anchor else "Other Models",
                    marker_color=colors[is_anchor],
                    orientation="h",
                    showlegend=True,
                    line=dict(width=2),
                    marker=dict(size=4, line=dict(width=1, color="white"))
                    if show_points
                    else None,
                    boxpoints="outliers" if show_points else False,
                )
            )

        # Add annotations for deltas
        if show_stats:
            annotations = []
            for _, row in model_stats.iterrows():
                if row[self.model_col] == self.anchor_model:
                    text = f"<b>{row['median']:.3f}</b> (baseline)"
                else:
                    delta_sign = "+" if row["delta_from_anchor"] > 0 else ""
                    text = (
                        f"{row['median']:.3f} "
                        f"({delta_sign}{row['delta_pct']:.1f}%)"
                    )

                annotations.append(
                    dict(
                        x=df[df[self.model_col] == row[self.model_col]][
                            plot_metric
                        ].max()
                        * 1.02,
                        y=row[self.model_col],
                        text=text,
                        showarrow=False,
                        xanchor="left",
                        font=dict(size=9, color="#333333"),
                    )
                )
            fig.update_layout(annotations=annotations)

        # Add vertical line at anchor median
        fig.add_vline(
            x=anchor_median,
            line_dash="dash",
            line_color="rgba(228, 87, 86, 0.3)",
            line_width=2,
            annotation_text=f"Anchor: {anchor_median:.3f}",
            annotation_position="top",
        )

        # Layout
        title_text = (
            title
            or f"{metric.upper()} Distribution vs Anchor ({self.anchor_model})"
        )
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                font=dict(size=16, color="#333333"),
            ),
            xaxis_title=metric.upper(),
            yaxis_title="Model",
            width=figsize[0],
            height=figsize[1],
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=100, r=150, t=80, b=60),
        )

        # Grid styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(128, 128, 128, 0.3)",
        )
        fig.update_yaxes(showgrid=False)

        # Print summary statistics
        if show_stats:
            print(f"\n{'='*60}")
            print(f"SUMMARY: {metric.upper()} vs {self.anchor_model}")
            print(f"{'='*60}")
            print(f"{'Model':<15} {'Median':>10} {'Delta':>10} {'Delta %':>10} {'N':>8}")
            print(f"{'-'*60}")
            for _, row in model_stats.iterrows():
                delta_sign = "+" if row["delta_from_anchor"] > 0 else ""
                print(
                    f"{row[self.model_col]:<15} {row['median']:>10.4f} "
                    f"{delta_sign}{row['delta_from_anchor']:>9.4f} "
                    f"{delta_sign}{row['delta_pct']:>9.1f}% {int(row['count']):>8}"
                )
            print(f"{'='*60}\n")

        return fig


__all__ = ["MetricsCalculator", "MetricResults"]
