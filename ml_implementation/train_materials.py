import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from ml_utils import load_clean_data, prepare_features

sat, _ = load_clean_data()

required_cols = [
    "dominant_material_fraction_1",
    "dominant_material_fraction_2",
    "dominant_material_fraction_3"
]

train_df = sat.dropna(subset=required_cols).copy()

features = [
    "launch_mass_kg", "launch_year",
    "perigee_km", "apogee_km", "inclination_deg",
    "operator", "country", "purpose", "orbit_type", "bus_family",
    "dry_mass_kg"
]

X = prepare_features(train_df[features])
y = train_df[required_cols]

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
model.fit(X, y)

joblib.dump(model, "ml/models/materials_model.pkl")
print("Saved materials_model.pkl!")