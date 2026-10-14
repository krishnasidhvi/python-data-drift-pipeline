from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("DataDriftPipeline.Ingestion")


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
