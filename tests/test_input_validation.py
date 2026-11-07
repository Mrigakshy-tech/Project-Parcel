import pandas as pd
import pytest
from src.parcel_utils import validate_input

def test_valid_input_keeps_rows_and_coerces_numeric():
    df = pd.DataFrame({"distance_km": ["10", "20"], "service_level": ["standard", "express"]})
    out = validate_input(df, ["distance_km", "service_level"], ["distance_km"])
    assert len(out) == 2
    assert out["distance_km"].dtype.kind in "fi"

def test_missing_required_column_fails_clearly():
    with pytest.raises(ValueError, match="Missing required"):
        validate_input(pd.DataFrame({"distance_km": [2]}), ["distance_km", "service_level"])

def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_input(pd.DataFrame(columns=["distance_km"]), ["distance_km"])

def test_infinite_numeric_value_is_rejected():
    with pytest.raises(ValueError, match="infinite"):
        validate_input(pd.DataFrame({"distance_km": [float("inf")]}), ["distance_km"], ["distance_km"])
