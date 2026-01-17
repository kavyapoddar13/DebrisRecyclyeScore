import joblib
import pandas as pd
from ml.ml_utils import prepare_features

dry_model = joblib.load("ml/models/dry_mass_model.pkl")
mat_model = joblib.load("ml/models/materials_model.pkl")

def predict_all(row_dict):
    df = pd.DataFrame([row_dict])

    # Predict dry mass first
    X_dry = prepare_features(df.copy())
    dry_mass = float(dry_model.predict(X_dry)[0])

    df["dry_mass_kg"] = dry_mass

    # Predict materials
    X_mat = prepare_features(df.copy())
    m1, m2, m3 = mat_model.predict(X_mat)[0]

    return {
        "dry_mass_kg": dry_mass,
        "dominant_material_fraction_1": float(m1),
        "dominant_material_fraction_2": float(m2),
        "dominant_material_fraction_3": float(m3)
    }
