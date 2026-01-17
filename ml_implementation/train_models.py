import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ml_utils import load_clean_data, prepare_features

sat, debris = load_clean_data()

# Only rows with known dry mass
train_df = sat[sat["dry_mass_kg"].notna()].copy()

features = [
    "launch_mass_kg", "launch_year",
    "perigee_km", "apogee_km", "inclination_deg",
    "operator", "country", "purpose", "orbit_type", "bus_family"
]

X = prepare_features(train_df[features])
y = train_df["dry_mass_kg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "ml/models/dry_mass_model.pkl")
print("Saved dry_mass_model.pkl!")