"""
Enterprise Client Production Integration Example.
Demonstrates how an enterprise data/ML platform integrates the SelfHealingMonitoringPipeline
into a batch scoring or Airflow/Prefect orchestrated job with automated alerting and retraining triggers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from drift import StatisticalDriftEngine
from ingestion import BatchIngestionGatekeeper
from pipeline import SelfHealingMonitoringPipeline
from schemas import CustomerFeatureRecord

logger = logging.getLogger("EnterpriseMLOps")


class EnterpriseDriftSentinel:
    """
    Drop-in client wrapper for enterprise pipelines.
    Automates baseline locking, batch scoring, alerts, and model retraining triggers.
    """

    def __init__(self, failure_threshold: float = 0.15):
        self.pipeline = SelfHealingMonitoringPipeline(
            gatekeeper=BatchIngestionGatekeeper(
                CustomerFeatureRecord, max_failure_rate=failure_threshold
            ),
            drift_engine=StatisticalDriftEngine(
                ks_alpha=0.05,
                psi_warning_threshold=0.10,
                psi_critical_threshold=0.25,
            ),
            numeric_features=["age", "annual_income", "credit_utilization", "monthly_spend"],
            categorical_features=["subscription_tier"],
        )

    def initialize_with_training_data(self, training_records: List[Dict[str, Any]]) -> None:
        """Step 1: Lock the baseline using the dataset used to train the current model."""
        logger.info(f"Locking baseline distribution across {len(training_records)} training records.")
        self.pipeline.fit_reference_baseline(training_records)

    def monitor_daily_batch(self, daily_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 2: Score the daily batch, isolate corrupt rows, and evaluate drift."""
        logger.info(f"Ingesting daily batch with {len(daily_records)} incoming records.")
        report = self.pipeline.process_production_batch(daily_records)

        # Step 3: Trigger enterprise actions based on the report
        if report["retraining_triggered"]:
            self._trigger_automated_retraining(report)
        elif any(f["severity"] == "WARNING" for f in report["feature_drift_summary"]):
            self._send_slack_investigation_alert(report)
        else:
            logger.info("Pipeline health verified: Distributions are STABLE.")

        # Step 4: Export visualization artifacts for MLOps dashboards
        self.pipeline.plot_dashboard(report, output_path="production_drift_dashboard.png")
        self.pipeline.export_html_report(report, output_path="production_drift_report.html")

        return report

    def _trigger_automated_retraining(self, report: Dict[str, Any]) -> None:
        """Webhook or trigger for Vertex AI / Kubeflow / Airflow retraining pipeline."""
        drifted_features = [
            f["feature_name"]
            for f in report["feature_drift_summary"]
            if f["severity"] == "CRITICAL_DRIFT"
        ]
        logger.warning(
            f"[ALERT] CRITICAL DRIFT DETECTED on {drifted_features}! "
            "Dispatching automated webhook to retrain ML model."
        )
        # Client can insert: requests.post("https://api.internal.mlops/retrain", json={...})

    def _send_slack_investigation_alert(self, report: Dict[str, Any]) -> None:
        """Sends non-blocking warning to data science team."""
        warning_features = [
            f["feature_name"]
            for f in report["feature_drift_summary"]
            if f["severity"] == "WARNING"
        ]
        logger.warning(
            f"[WARNING] Moderate drift detected on {warning_features}. Alerting MLOps oncall channel."
        )


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    rng = np.random.default_rng(101)

    # 1. Simulate training baseline
    train_data = [
        {
            "record_id": f"TRAIN-{i:04d}",
            "age": int(np.clip(rng.normal(36, 7), 18, 90)),
            "annual_income": float(np.round(rng.lognormal(mean=11.1, sigma=0.4), 2)),
            "credit_utilization": float(np.clip(rng.beta(2, 5), 0.0, 1.0)),
            "monthly_spend": float(np.round(rng.normal(1900, 350), 2)),
            "subscription_tier": rng.choice(["free", "pro", "enterprise"], p=[0.6, 0.3, 0.1]),
        }
        for i in range(1000)
    ]

    sentinel = EnterpriseDriftSentinel()
    sentinel.initialize_with_training_data(train_data)

    # 2. Simulate daily production batch with drift
    daily_data = [
        {
            "record_id": f"DAY-{i:04d}",
            "age": int(np.clip(rng.normal(36, 7), 18, 90)),
            "annual_income": float(np.round(rng.lognormal(mean=11.6, sigma=0.4), 2)),  # Shifted
            "credit_utilization": float(np.clip(rng.beta(2, 5), 0.0, 1.0)),
            "monthly_spend": float(np.round(rng.normal(2600, 450), 2)),  # Shifted
            "subscription_tier": rng.choice(["free", "pro", "enterprise"], p=[0.4, 0.4, 0.2]),
        }
        for i in range(1000)
    ]

    # Inject 2 corrupted rows to verify DLQ
    daily_data.append({"record_id": "ERR-1", "age": 12, "annual_income": 50000, "credit_utilization": 0.2, "monthly_spend": 100, "subscription_tier": "free"})

    report = sentinel.monitor_daily_batch(daily_data)
    print("\nClient batch scoring completed successfully. Retrain Triggered:", report["retraining_triggered"])
