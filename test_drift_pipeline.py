from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from schemas import CustomerFeatureRecord, DriftSeverity
from ingestion import BatchIngestionGatekeeper, QuarantinedRecord
from drift import FeatureDriftMetrics, StatisticalDriftEngine
from pipeline import SelfHealingMonitoringPipeline


class TestCustomerFeatureRecord(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "record_id": "REC-12345",
            "age": 35,
            "annual_income": 85000.0,
            "credit_utilization": 0.25,
            "monthly_spend": 2000.0,
            "subscription_tier": "pro",
        }

    def test_valid_record(self):
        record = CustomerFeatureRecord.model_validate(self.valid_data)
        self.assertEqual(record.record_id, "REC-12345")
        self.assertEqual(record.age, 35)
        self.assertEqual(record.subscription_tier, "pro")

    def test_underage_rejected(self):
        data = self.valid_data.copy()
        data["age"] = 17
        with self.assertRaises(ValidationError) as ctx:
            CustomerFeatureRecord.model_validate(data)
        self.assertIn("age", str(ctx.exception))

    def test_overage_rejected(self):
        data = self.valid_data.copy()
        data["age"] = 115
        with self.assertRaises(ValidationError):
            CustomerFeatureRecord.model_validate(data)

    def test_invalid_tier_rejected(self):
        data = self.valid_data.copy()
        data["subscription_tier"] = "diamond"
        with self.assertRaises(ValidationError):
            CustomerFeatureRecord.model_validate(data)

    def test_extra_field_forbidden(self):
        data = self.valid_data.copy()
        data["unexpected_column"] = 123
        with self.assertRaises(ValidationError):
            CustomerFeatureRecord.model_validate(data)

    def test_nan_or_inf_rejected(self):
        data_nan = self.valid_data.copy()
        data_nan["annual_income"] = float("nan")
        with self.assertRaises(ValidationError):
            CustomerFeatureRecord.model_validate(data_nan)

        data_inf = self.valid_data.copy()
        data_inf["credit_utilization"] = float("inf")
        with self.assertRaises(ValidationError):
            CustomerFeatureRecord.model_validate(data_inf)

    def test_spend_to_income_sanity_check(self):
        # Monthly spend = 50,000 * 12 = 600,000 > 100,000 * 5 = 500,000
        data = self.valid_data.copy()
        data["annual_income"] = 100000.0
        data["monthly_spend"] = 50000.0
        with self.assertRaises(ValidationError) as ctx:
            CustomerFeatureRecord.model_validate(data)
        self.assertIn("Anomalous spend-to-income ratio", str(ctx.exception))


class TestBatchIngestionGatekeeper(unittest.TestCase):
    def setUp(self):
        self.gatekeeper = BatchIngestionGatekeeper(
            CustomerFeatureRecord, max_failure_rate=0.25
        )
        self.valid_sample = {
            "record_id": "REC-001",
            "age": 30,
            "annual_income": 60000.0,
            "credit_utilization": 0.15,
            "monthly_spend": 1000.0,
            "subscription_tier": "free",
        }

    def test_empty_batch_raises_error(self):
        with self.assertRaises(ValueError):
            self.gatekeeper.ingest_and_validate([])

    def test_dlq_routing(self):
        batch = [
            self.valid_sample,
            {**self.valid_sample, "record_id": "REC-002"},
            {**self.valid_sample, "record_id": "REC-003"},
            {**self.valid_sample, "record_id": "REC-004"},
            {"record_id": "BAD-01", "age": 12, "annual_income": 1000.0, "credit_utilization": 0.1, "monthly_spend": 10.0, "subscription_tier": "free"},
        ]
        valid_df, dlq = self.gatekeeper.ingest_and_validate(batch)
        self.assertEqual(len(valid_df), 4)
        self.assertEqual(len(dlq), 1)
        self.assertIsInstance(dlq[0], QuarantinedRecord)
        self.assertEqual(dlq[0].raw_payload["record_id"], "BAD-01")

    def test_circuit_breaker_triggered(self):
        bad_records = [
            {"record_id": f"BAD-{i}", "age": 10, "annual_income": -100, "credit_utilization": 5.0, "monthly_spend": 10.0, "subscription_tier": "unknown"}
            for i in range(10)
        ]
        batch = [self.valid_sample] + bad_records  # 10 / 11 > 90% failure
        with self.assertRaises(RuntimeError) as ctx:
            self.gatekeeper.ingest_and_validate(batch)
        self.assertIn("Circuit Breaker Triggered", str(ctx.exception))


class TestStatisticalDriftEngine(unittest.TestCase):
    def setUp(self):
        self.engine = StatisticalDriftEngine(
            ks_alpha=0.05,
            psi_warning_threshold=0.10,
            psi_critical_threshold=0.25,
            num_bins=10,
        )
        self.rng = np.random.default_rng(42)

    def test_numeric_psi_stable(self):
        ref = self.rng.normal(50, 10, size=2000)
        curr = self.rng.normal(50, 10, size=2000)
        psi = self.engine.compute_numeric_psi(ref, curr)
        self.assertLess(psi, 0.05)

    def test_numeric_psi_drift(self):
        ref = self.rng.normal(50, 10, size=2000)
        curr = self.rng.normal(70, 15, size=2000)
        psi = self.engine.compute_numeric_psi(ref, curr)
        self.assertGreater(psi, 0.25)

    def test_categorical_psi(self):
        ref = pd.Series(["A"] * 600 + ["B"] * 300 + ["C"] * 100)
        curr_stable = pd.Series(["A"] * 605 + ["B"] * 295 + ["C"] * 100)
        psi_stable = self.engine.compute_categorical_psi(ref, curr_stable)
        self.assertLess(psi_stable, 0.05)

        curr_drift = pd.Series(["A"] * 200 + ["B"] * 400 + ["C"] * 400)
        psi_drift = self.engine.compute_categorical_psi(ref, curr_drift)
        self.assertGreater(psi_drift, 0.25)

    def test_evaluate_feature_numeric_stable_and_drift(self):
        ref_df = pd.DataFrame({
            "age": self.rng.normal(40, 5, 1000),
            "income": self.rng.normal(50000, 5000, 1000),
        })
        curr_df = pd.DataFrame({
            "age": self.rng.normal(40, 5, 1000),  # stable
            "income": self.rng.normal(80000, 10000, 1000),  # severe shift
        })

        age_metrics = self.engine.evaluate_feature(ref_df, curr_df, "age", is_categorical=False)
        self.assertEqual(age_metrics.severity, DriftSeverity.STABLE)
        self.assertFalse(age_metrics.drift_detected)

        income_metrics = self.engine.evaluate_feature(ref_df, curr_df, "income", is_categorical=False)
        self.assertIn(income_metrics.severity, [DriftSeverity.WARNING, DriftSeverity.CRITICAL_DRIFT])
        self.assertTrue(income_metrics.drift_detected)


class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.gatekeeper = BatchIngestionGatekeeper(CustomerFeatureRecord, max_failure_rate=0.20)
        self.engine = StatisticalDriftEngine()
        self.pipeline = SelfHealingMonitoringPipeline(
            gatekeeper=self.gatekeeper,
            drift_engine=self.engine,
            numeric_features=["age", "annual_income"],
            categorical_features=["subscription_tier"],
        )
        self.rng = np.random.default_rng(123)

    def test_unfitted_pipeline_raises(self):
        with self.assertRaises(RuntimeError):
            self.pipeline.process_production_batch([{"record_id": "test"}])

    def test_end_to_end_pipeline(self):
        ref_data = [
            {
                "record_id": f"REF-{i}",
                "age": int(self.rng.integers(20, 60)),
                "annual_income": float(self.rng.normal(60000, 5000)),
                "credit_utilization": 0.2,
                "monthly_spend": 1000.0,
                "subscription_tier": "free",
            }
            for i in range(500)
        ]
        self.pipeline.fit_reference_baseline(ref_data)

        # Shift income dramatically in production
        prod_data = [
            {
                "record_id": f"PROD-{i}",
                "age": int(self.rng.integers(20, 60)),
                "annual_income": float(self.rng.normal(120000, 10000)),
                "credit_utilization": 0.2,
                "monthly_spend": 1000.0,
                "subscription_tier": "free",
            }
            for i in range(500)
        ]

        report = self.pipeline.process_production_batch(prod_data)
        self.assertEqual(report["valid_record_count"], 500)
        self.assertEqual(report["quarantined_record_count"], 0)
        self.assertTrue(report["retraining_triggered"])
        self.assertEqual(len(report["feature_drift_summary"]), 3)

        # Test visualization generation
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_dashboard.png")
            html_path = os.path.join(tmpdir, "test_report.html")

            self.pipeline.plot_dashboard(report, output_path=img_path)
            self.pipeline.export_html_report(report, output_path=html_path)

            self.assertTrue(os.path.exists(img_path))
            self.assertGreater(os.path.getsize(img_path), 1000)
            self.assertTrue(os.path.exists(html_path))
            self.assertGreater(os.path.getsize(html_path), 500)


if __name__ == "__main__":
    unittest.main()

