from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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
