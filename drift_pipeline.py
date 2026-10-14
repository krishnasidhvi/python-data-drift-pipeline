from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy import stats

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("DataDriftPipeline")


# =====================================================================
# MODULE 1: PYDANTIC V2 SCHEMA CONTRACT
# =====================================================================
class DriftSeverity(str, Enum):
    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"

    def __str__(self) -> str:
        return self.value


class CustomerFeatureRecord(BaseModel):
    """
    Row-level data contract using Pydantic v2.
    Enforces strict boundary constraints and cross-field business logic.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unexpected upstream columns
        str_strip_whitespace=True,  # Clean string inputs automatically
        validate_assignment=True,
    )

    record_id: Annotated[str, Field(min_length=3, max_length=64)]
    event_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    age: Annotated[int, Field(ge=18, le=110, description="Customer age in years")]
    annual_income: Annotated[float, Field(gt=0.0, le=5e7)]
    credit_utilization: Annotated[float, Field(ge=0.0, le=1.0)]
    monthly_spend: Annotated[float, Field(ge=0.0)]
    subscription_tier: Literal["free", "pro", "enterprise"]

    @field_validator("annual_income", "monthly_spend", "credit_utilization")
    @classmethod
    def reject_nan_or_inf(cls, value: float) -> float:
        """Prevent NaN or Inf floats from corrupting downstream numpy math."""
        if np.isnan(value) or np.isinf(value):
            raise ValueError("Numeric feature cannot be NaN or Infinite.")
        return value

    @model_validator(mode="after")
    def check_spend_to_income_sanity(self) -> CustomerFeatureRecord:
        """Cross-field invariant: Annualized spend cannot exceed 5x annual income."""
        if (self.monthly_spend * 12) > (self.annual_income * 5.0):
            raise ValueError(
                f"Anomalous spend-to-income ratio: monthly_spend={self.monthly_spend}, "
                f"annual_income={self.annual_income}"
            )
        return self


# =====================================================================
# MODULE 2: INGESTION & DEAD-LETTER QUEUE (DLQ) GATEKEEPER
# =====================================================================
@dataclass
class QuarantinedRecord:
    raw_payload: Dict[str, Any]
    error_details: List[Dict[str, Any]]
    quarantined_at: str


class BatchIngestionGatekeeper:
    """
    Validates raw dictionary payloads against a Pydantic schema.
    Routes valid records to a Pandas DataFrame and invalid rows to a DLQ.
    """

    def __init__(self, schema_cls: type[BaseModel], max_failure_rate: float = 0.20):
        self.schema_cls = schema_cls
        self.max_failure_rate = max_failure_rate

    def ingest_and_validate(
        self, raw_records: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, List[QuarantinedRecord]]:
        if not raw_records:
            raise ValueError("Received empty batch for ingestion.")

        valid_rows: List[Dict[str, Any]] = []
        dlq_rows: List[QuarantinedRecord] = []

        for row in raw_records:
            try:
                validated_obj = self.schema_cls.model_validate(row)
                valid_rows.append(validated_obj.model_dump())
            except ValidationError as exc:
                dlq_rows.append(
                    QuarantinedRecord(
                        raw_payload=row,
                        error_details=exc.errors(include_url=False),
                        quarantined_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

        total = len(raw_records)
        failure_rate = len(dlq_rows) / total
        logger.info(
            f"Validation Complete: {len(valid_rows)}/{total} passed "
            f"({len(dlq_rows)} routed to DLQ | Failure rate: {failure_rate:.2%})"
        )

        # Circuit breaker if upstream schema breaks completely
        if failure_rate > self.max_failure_rate:
            raise RuntimeError(
                f"Circuit Breaker Triggered: Batch failure rate ({failure_rate:.2%}) "
                f"exceeded threshold ({self.max_failure_rate:.2%})."
            )

        return pd.DataFrame(valid_rows), dlq_rows


# =====================================================================
# MODULE 3: STATISTICAL DRIFT ENGINE (KS-TEST & PSI)
# =====================================================================
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


# =====================================================================
# MODULE 4: END-TO-END PIPELINE ORCHESTRATOR
# =====================================================================
class SelfHealingMonitoringPipeline:
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



# =====================================================================
# EXECUTION DEMO: SIMULATING SCHEMA VIOLATIONS + COVARIATE DRIFT
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    N = 2000

    # 1. Generate clean Reference Baseline (Training Distribution)
    ref_payload = [
        {
            "record_id": f"REF-{i:05d}",
            "age": int(np.clip(rng.normal(35, 8), 18, 90)),
            "annual_income": float(np.round(rng.lognormal(mean=11.0, sigma=0.4), 2)),
            "credit_utilization": float(np.clip(rng.beta(2, 5), 0.0, 1.0)),
            "monthly_spend": float(np.round(rng.normal(1800, 400), 2)),
            "subscription_tier": rng.choice(
                ["free", "pro", "enterprise"], p=[0.60, 0.30, 0.10]
            ),
        }
        for i in range(N)
    ]

    # 2. Generate Production Batch with:
    # - Stable 'age' (no drift)
    # - Moderate drift on 'credit_utilization'
    # - Severe macroeconomic shift on 'annual_income' & 'monthly_spend'
    # - Injected malformed records to test Pydantic DLQ
    prod_payload = [
        {
            "record_id": f"PROD-{i:05d}",
            "age": int(np.clip(rng.normal(35.2, 8.1), 18, 90)),  # Stable
            "annual_income": float(np.round(rng.lognormal(mean=11.4, sigma=0.5), 2)),  # Severe Shift
            "credit_utilization": float(np.clip(rng.beta(2.6, 4.5), 0.0, 1.0)),  # Moderate Shift
            "monthly_spend": float(np.round(rng.normal(2450, 550), 2)),  # Severe Shift
            "subscription_tier": rng.choice(
                ["free", "pro", "enterprise"], p=[0.35, 0.45, 0.20]  # Category Shift
            ),
        }
        for i in range(N)
    ]

    # Inject 3 corrupted rows that violate Pydantic constraints
    prod_payload.extend(
        [
            {
                "record_id": "BAD-01",
                "age": 14,
                "annual_income": 50000,
                "credit_utilization": 0.3,
                "monthly_spend": 1000,
                "subscription_tier": "free",
            },  # Underage (<18)
            {
                "record_id": "BAD-02",
                "age": 40,
                "annual_income": 20000,
                "credit_utilization": 0.5,
                "monthly_spend": 15000,
                "subscription_tier": "pro",
            },  # Violates cross-field spend/income sanity check
            {
                "record_id": "BAD-03",
                "age": 29,
                "annual_income": 75000,
                "credit_utilization": 1.4,
                "monthly_spend": 1200,
                "subscription_tier": "gold",
            },  # Invalid utilization (>1.0) & tier literal
        ]
    )

    # 3. Initialize & Run Pipeline
    pipeline = SelfHealingMonitoringPipeline(
        gatekeeper=BatchIngestionGatekeeper(CustomerFeatureRecord, max_failure_rate=0.10),
        drift_engine=StatisticalDriftEngine(
            ks_alpha=0.05, psi_warning_threshold=0.10, psi_critical_threshold=0.25
        ),
        numeric_features=["age", "annual_income", "credit_utilization", "monthly_spend"],
        categorical_features=["subscription_tier"],
    )

    pipeline.fit_reference_baseline(ref_payload)
    report = pipeline.process_production_batch(prod_payload)

    # 4. Display Summary Table
    print("\n" + "=" * 82)
    print(f"PIPELINE HEALTH REPORT | Retraining Triggered: {report['retraining_triggered']}")
    print(
        f"Valid Rows: {report['valid_record_count']} | Quarantined (DLQ): {report['quarantined_record_count']}"
    )
    print("=" * 82)
    summary_df = pd.DataFrame(report["feature_drift_summary"])
    print(summary_df.to_string(index=False))

    # 5. Generate Visualizations & Interactive Reports
    dashboard_img = pipeline.plot_dashboard(report, output_path="drift_dashboard.png")
    html_report = pipeline.export_html_report(report, output_path="drift_report.html")
    print("\n" + "=" * 82)
    print("VISUALIZATION METRICS GENERATED:")
    print(f"  -> Static High-Resolution Dashboard: {dashboard_img}")
    print(f"  -> Interactive Standalone HTML Report: {html_report}")
    print("=" * 82)

