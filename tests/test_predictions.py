import numpy as np
import pandas as pd
from src.parcel_utils import predict_risk

class DummyClassifier:
    def predict_proba(self, X):
        p = np.array([0.8, 0.2])[:len(X)]
        return np.column_stack([1-p, p])

def test_prediction_schema_and_range():
    X = pd.DataFrame({"distance_km": [10, 20]})
    out = predict_risk(DummyClassifier(), X, ["distance_km"], ["distance_km"])
    assert list(out.columns) == ["late_risk", "predicted_late"]
    assert out["late_risk"].between(0, 1).all()
    assert out["predicted_late"].tolist() == [1, 0]
