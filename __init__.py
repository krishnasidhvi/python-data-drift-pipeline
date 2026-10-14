"""
Modular Data Ingestion and Statistical Drift Detection Pipeline.
Combines Pydantic v2 data contract enforcement with KS-test and PSI statistical monitors.
"""

from schemas import CustomerFeatureRecord, DriftSeverity
from ingestion import BatchIngestionGatekeeper, QuarantinedRecord
from drift import FeatureDriftMetrics, StatisticalDriftEngine
from pipeline import SelfHealingMonitoringPipeline
from visualization import plot_drift_dashboard, generate_html_report

__all__ = [
    "CustomerFeatureRecord",
    "DriftSeverity",
    "BatchIngestionGatekeeper",
    "QuarantinedRecord",
    "FeatureDriftMetrics",
    "StatisticalDriftEngine",
    "SelfHealingMonitoringPipeline",
    "plot_drift_dashboard",
    "generate_html_report",
]

