import pandas as pd

def load_clean_data():
    sat = pd.read_csv("data/norad_mass_dataset_500.csv")
    debris = pd.read_csv("data/active_debris_updated.csv")
    return sat, debris

def encode_categories(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df

def prepare_features(df):
    df = df.copy()

    # Convert numeric columns from strings with commas -> float
    numeric_cols = [
        "launch_mass_kg", "launch_year",
        "perigee_km", "apogee_km", "inclination_deg",
        "dry_mass_kg"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("", None)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode categorical columns
    categorical_cols = ["operator", "country", "purpose", "orbit_type", "bus_family"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Fill missing numerics
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df