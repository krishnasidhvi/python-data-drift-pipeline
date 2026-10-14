from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from schemas import DriftSeverity


@dataclass
class FeatureDriftMetrics:
    feature_name: str
    feature_type: str
    ks_statistic: float | None
    ks_p_value: float | None
    psi_score: float
    severity: DriftSeverity
    drift_detected: bool


class StatisticalDriftEngine:
    """
    Computes Two-Sample Kolmogorov-Smirnov (KS) and Population Stability Index (PSI)
    between a reference baseline distribution and a current production batch.
    """

    def __init__(
        self,
        ks_alpha: float = 0.05,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ):
        self.ks_alpha = ks_alpha
        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.num_bins = num_bins
        self.epsilon = epsilon  # Laplace smoothing to avoid log(0) or div-by-zero

    def compute_numeric_psi(
        self, ref_series: np.ndarray, curr_series: np.ndarray
    ) -> float:
        """
        Calculates PSI using reference-quantile binning so bins have equal
        expected probability mass on the baseline distribution.
        """
        quantiles = np.linspace(0, 100, self.num_bins + 1)
        bin_edges = np.percentile(ref_series, quantiles)

        # Deduplicate edges in case of heavy ties/zero-inflation
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Extend outer edges to -inf and +inf so out-of-range production values are captured
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_series, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_series, bins=bin_edges)

        # Convert to proportions with Laplace smoothing
        ref_props = np.where(ref_counts == 0, self.epsilon, ref_counts / len(ref_series))
        curr_props = np.where(curr_counts == 0, self.epsilon, curr_counts / len(curr_series))

        # Normalize to ensure exact probability distributions
        ref_props = ref_props / np.sum(ref_props)
        curr_props = curr_props / np.sum(curr_props)

        psi_value = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        return float(np.round(psi_value, 5))

    def compute_categorical_psi(
        self, ref_series: pd.Series, curr_series: pd.Series
    ) -> float:
        """Calculates PSI across discrete category levels."""
        all_categories = sorted(set(ref_series.unique()).union(set(curr_series.unique())))

        ref_freqs = ref_series.value_counts(normalize=True).to_dict()
        curr_freqs = curr_series.value_counts(normalize=True).to_dict()

        ref_props = np.array([ref_freqs.get(cat, self.epsilon) for cat in all_categories])
        curr_props = np.array([curr_freqs.get(cat, self.epsilon) for cat in all_categories])

        ref_props /= np.sum(ref_props)
        curr_props /= np.sum(curr_props)

        psi_value = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        return float(np.round(psi_value, 5))

    def evaluate_feature(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        feature: str,
        is_categorical: bool = False,
    ) -> FeatureDriftMetrics:
        if is_categorical:
            psi = self.compute_categorical_psi(ref_df[feature], curr_df[feature])
            ks_stat, ks_pval = None, None
        else:
            ref_arr = ref_df[feature].dropna().to_numpy(dtype=float)
            curr_arr = curr_df[feature].dropna().to_numpy(dtype=float)

            # Two-sample KS test
            ks_result = stats.ks_2samp(ref_arr, curr_arr, method="asymp")
            ks_stat = float(np.round(ks_result.statistic, 5))
            ks_pval = float(np.round(ks_result.pvalue, 6))

            psi = self.compute_numeric_psi(ref_arr, curr_arr)

        # Adjudicate severity using both PSI and KS p-value
        if psi >= self.psi_critical or (
            ks_pval is not None and ks_pval < (self.ks_alpha / 10) and psi >= self.psi_warning
        ):
            severity = DriftSeverity.CRITICAL_DRIFT
            drift_detected = True
        elif psi >= self.psi_warning or (ks_pval is not None and ks_pval < self.ks_alpha):
            severity = DriftSeverity.WARNING
            drift_detected = True
        else:
            severity = DriftSeverity.STABLE
            drift_detected = False

        return FeatureDriftMetrics(
            feature_name=feature,
            feature_type="categorical" if is_categorical else "numeric",
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            psi_score=psi,
            severity=severity,
            drift_detected=drift_detected,
        )
