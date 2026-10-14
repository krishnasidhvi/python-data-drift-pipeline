from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from drift import FeatureDriftMetrics, StatisticalDriftEngine
from ingestion import BatchIngestionGatekeeper
from schemas import DriftSeverity

logger = logging.getLogger("DataDriftPipeline.Pipeline")


class SelfHealingMonitoringPipeline:
    """
    Decoupled four-stage pipeline orchestrator:
    1. Schema Gatekeeper (Pydantic v2 + DLQ routing)
    2. Vectorized DataFrame generation
    3. Dual Drift Engine (KS-test + quantile-binned PSI)
    4. Actionable Drift Reporting & Retraining trigger
    """

    def __init__(
        self,
        gatekeeper: BatchIngestionGatekeeper,
        drift_engine: StatisticalDriftEngine,
        numeric_features: List[str],
        categorical_features: List[str],
    ):
        self.gatekeeper = gatekeeper
        self.drift_engine = drift_engine
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.reference_df: pd.DataFrame | None = None

    def fit_reference_baseline(self, raw_reference_data: List[Dict[str, Any]]) -> None:
        logger.info("Ingesting and locking Reference Baseline distribution...")
        valid_df, _ = self.gatekeeper.ingest_and_validate(raw_reference_data)
        self.reference_df = valid_df

    def process_production_batch(
        self, raw_production_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if self.reference_df is None:
            raise RuntimeError("Reference baseline must be fitted before scoring batches.")

        logger.info("Processing incoming Production Batch...")
        curr_df, dlq_records = self.gatekeeper.ingest_and_validate(raw_production_data)

        feature_reports: List[FeatureDriftMetrics] = []

        for feat in self.numeric_features:
            feature_reports.append(
                self.drift_engine.evaluate_feature(
                    self.reference_df, curr_df, feat, is_categorical=False
                )
            )

        for feat in self.categorical_features:
            feature_reports.append(
                self.drift_engine.evaluate_feature(
                    self.reference_df, curr_df, feat, is_categorical=True
                )
            )

        retrain_required = any(
            r.severity == DriftSeverity.CRITICAL_DRIFT for r in feature_reports
        )

        self.current_df = curr_df

        return {
            "batch_timestamp": datetime.now(timezone.utc).isoformat(),
            "valid_record_count": len(curr_df),
            "quarantined_record_count": len(dlq_records),
            "retraining_triggered": retrain_required,
            "feature_drift_summary": [asdict(r) for r in feature_reports],
            "sample_dlq_errors": [asdict(d) for d in dlq_records[:2]],
        }

    def plot_dashboard(self, report: Dict[str, Any], output_path: str = "drift_dashboard.png") -> str:
        """Generates a publication-grade multi-panel PNG drift dashboard."""
        if self.reference_df is None or getattr(self, "current_df", None) is None:
            raise RuntimeError("Pipeline must process a production batch before plotting.")
        from visualization import plot_drift_dashboard
        return plot_drift_dashboard(report, self.reference_df, self.current_df, output_path)

    def export_html_report(self, report: Dict[str, Any], output_path: str = "drift_report.html") -> str:
        """Exports an interactive, self-contained HTML monitoring dashboard."""
        if self.reference_df is None or getattr(self, "current_df", None) is None:
            raise RuntimeError("Pipeline must process a production batch before exporting HTML report.")
        from visualization import generate_html_report
        return generate_html_report(report, self.reference_df, self.current_df, output_path)

