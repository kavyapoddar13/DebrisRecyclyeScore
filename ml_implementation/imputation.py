import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

INPUT_FILE = "norad_mass_dataset_500.xlsx"
OUTPUT_FILE = "norad_mass_dataset_500_imputed.xlsx"

TEMPLATE_MIN_SIZE = 3
EPS = 1e-9

BASE_CATEGORICAL_FEATURES = ["purpose", "bus_family", "operator", "country", "orbit_type"]
BASE_NUMERIC_FEATURES = ["launch_mass_kg", "launch_year", "perigee_km", "apogee_km", "inclination_deg"]

def clean_numeric_series(s):
    """
    Convert a pandas Series to numeric robustly:
    - remove commas and whitespace
    - replace empty strings with NaN
    - coerce errors to NaN
    """
    if s.dtype == object:
        s_clean = s.astype(str).str.replace(",", "", regex=False).str.strip()
        s_clean = s_clean.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
        return pd.to_numeric(s_clean, errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")

def normalize_three(f1, f2, f3):
    arr = np.array([
        float(f1) if pd.notna(f1) else 0.0,
        float(f2) if pd.notna(f2) else 0.0,
        float(f3) if pd.notna(f3) else 0.0
    ], dtype=float)
    total = arr.sum()
    if total <= EPS:
        return 1/3, 1/3, 1/3
    return tuple((arr / total).tolist())

df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

expected_cols = ["object_id", "satellite_name", "launch_year", "launch_mass_kg", "dry_mass_kg",
                 "operator", "country", "purpose", "orbit_type", "perigee_km", "apogee_km",
                 "inclination_deg", "bus_family",
                 "dominant_material_1","dominant_material_fraction_1",
                 "dominant_material_2","dominant_material_fraction_2",
                 "dominant_material_3","dominant_material_fraction_3"]
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan

numeric_cols_to_clean = set(BASE_NUMERIC_FEATURES + ["dry_mass_kg", "launch_mass_kg",
                                                     "dominant_material_fraction_1",
                                                     "dominant_material_fraction_2",
                                                     "dominant_material_fraction_3"])
for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = clean_numeric_series(df[col])
    else:
        df[col] = np.nan

if "launch_year" in df.columns:
    df["launch_year"] = clean_numeric_series(df["launch_year"]).astype(pd.Int64Dtype())

for cat in BASE_CATEGORICAL_FEATURES + ["bus_family"]:
    if cat not in df.columns:
        df[cat] = "UNKNOWN"
    else:
        df[cat] = df[cat].fillna("UNKNOWN").astype(str)

mask_train_dry = df["dry_mass_kg"].notna() & df["launch_mass_kg"].notna()
mask_pred_dry = df["dry_mass_kg"].isna() & df["launch_mass_kg"].notna()

features = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES

for num in BASE_NUMERIC_FEATURES:
    if num not in df.columns:
        df[num] = np.nan
    else:
        df[num] = clean_numeric_series(df[num])
        
features_df = df[features].copy()

for num in BASE_NUMERIC_FEATURES:
    median_val = features_df[num].median(skipna=True)
    if pd.isna(median_val):
        median_val = 0.0
    features_df[num] = features_df[num].fillna(median_val)
    
cat_df = pd.get_dummies(features_df[BASE_CATEGORICAL_FEATURES].astype(str), dummy_na=True)
X_all = pd.concat([features_df[BASE_NUMERIC_FEATURES].reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

X_train = X_all.loc[mask_train_dry].copy()
y_train = df.loc[mask_train_dry, "dry_mass_kg"].astype(float)
X_pred = X_all.loc[mask_pred_dry].copy()

if len(X_train) >= 10:
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_reg.fit(X_train, y_train)
    dry_pred = rf_reg.predict(X_pred)
else:
    ratios = (df.loc[mask_train_dry, "dry_mass_kg"] / df.loc[mask_train_dry, "launch_mass_kg"]).replace([np.inf, -np.inf], np.nan).dropna()
    median_ratio = ratios.median() if len(ratios) > 0 else 0.5
    dry_pred = df.loc[mask_pred_dry, "launch_mass_kg"].astype(float) * median_ratio

pred_indices = list(X_pred.index)
dry_pred_clipped = []
for idx, pred in zip(pred_indices, dry_pred):
    launch_mass = df.at[idx, "launch_mass_kg"]
    if pd.isna(launch_mass):
        dry_pred_clipped.append(np.nan)
        continue
    if launch_mass <= 20:
        val = float(launch_mass) * 0.98
    else:
        lower = 0.2 * launch_mass
        upper = 0.95 * launch_mass
        try:
            val = float(pred)
        except Exception:
            val = np.nan
        if not np.isfinite(val):
            ratios = (df.loc[mask_train_dry, "dry_mass_kg"] / df.loc[mask_train_dry, "launch_mass_kg"]).replace([np.inf, -np.inf], np.nan).dropna()
            median_ratio = ratios.median() if len(ratios)>0 else 0.5
            val = launch_mass * median_ratio
        val = max(lower, min(upper, val))
    dry_pred_clipped.append(val)

df.loc[mask_pred_dry, "dry_mass_kg"] = dry_pred_clipped

group_cols = ["bus_family", "purpose"]
for g in group_cols:
    if g not in df.columns:
        df[g] = "UNKNOWN"

grouped = df.groupby(group_cols)

template_mode = {}
template_mean_frac = {}
for name, gdf in grouped:
    if len(gdf) < TEMPLATE_MIN_SIZE:
        continue
    
    def safe_mode(series):
        m = series.dropna().mode()
        return m.iloc[0] if len(m) > 0 else None
    m1 = safe_mode(gdf["dominant_material_1"]) if "dominant_material_1" in gdf.columns else None
    m2 = safe_mode(gdf["dominant_material_2"]) if "dominant_material_2" in gdf.columns else None
    m3 = safe_mode(gdf["dominant_material_3"]) if "dominant_material_3" in gdf.columns else None
    f1 = gdf["dominant_material_fraction_1"].mean(skipna=True) if "dominant_material_fraction_1" in gdf.columns else np.nan
    f2 = gdf["dominant_material_fraction_2"].mean(skipna=True) if "dominant_material_fraction_2" in gdf.columns else np.nan
    f3 = gdf["dominant_material_fraction_3"].mean(skipna=True) if "dominant_material_fraction_3" in gdf.columns else np.nan
    template_mode[name] = (m1, m2, m3)
    template_mean_frac[name] = (f1, f2, f3)

def apply_template(row):
    key = (row.get("bus_family", "UNKNOWN"), row.get("purpose", "UNKNOWN"))
    return template_mode.get(key, (None, None, None)), template_mean_frac.get(key, (np.nan, np.nan, np.nan))

missing_mat_mask = df["dominant_material_1"].isna() | df["dominant_material_2"].isna() | df["dominant_material_3"].isna()
for idx in df[missing_mat_mask].index:
    (m1_tpl, m2_tpl, m3_tpl), _ = apply_template(df.loc[idx])
    if m1_tpl is not None:
        if pd.isna(df.at[idx, "dominant_material_1"]) and m1_tpl is not None:
            df.at[idx, "dominant_material_1"] = m1_tpl
        if pd.isna(df.at[idx, "dominant_material_2"]) and m2_tpl is not None:
            df.at[idx, "dominant_material_2"] = m2_tpl
        if pd.isna(df.at[idx, "dominant_material_3"]) and m3_tpl is not None:
            df.at[idx, "dominant_material_3"] = m3_tpl

def train_and_predict_material(slot_name):
    mask_known = df[slot_name].notna()
    mask_unknown = df[slot_name].isna()
    if mask_unknown.sum() == 0:
        return
    if mask_known.sum() < 10:
        global_mode = df.loc[mask_known, slot_name].dropna().mode()
        fill_value = global_mode.iloc[0] if len(global_mode) > 0 else "Unknown"
        df.loc[mask_unknown, slot_name] = fill_value
        return

    feature_cols = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + ["dry_mass_kg"]
    feat_df = df[feature_cols].copy()
    for num in BASE_NUMERIC_FEATURES + ["dry_mass_kg"]:
        if num in feat_df.columns:
            med = feat_df[num].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            feat_df[num] = feat_df[num].fillna(med)

    cat_df = pd.get_dummies(feat_df[BASE_CATEGORICAL_FEATURES].astype(str), dummy_na=True)
    X_all_mat = pd.concat([feat_df[BASE_NUMERIC_FEATURES + ["dry_mass_kg"]].reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

    X_train = X_all_mat.loc[mask_known].copy()
    y_train = df.loc[mask_known, slot_name].astype(str)
    X_pred = X_all_mat.loc[mask_unknown].copy()

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_pred)
    df.loc[mask_unknown, slot_name] = preds

train_and_predict_material("dominant_material_1")
train_and_predict_material("dominant_material_2")
train_and_predict_material("dominant_material_3")

mask_frac_missing = df["dominant_material_fraction_1"].isna() | df["dominant_material_fraction_2"].isna() | df["dominant_material_fraction_3"].isna()
for idx in df[mask_frac_missing].index:
    key = (df.at[idx, "bus_family"], df.at[idx, "purpose"])
    f1_tpl, f2_tpl, f3_tpl = template_mean_frac.get(key, (np.nan, np.nan, np.nan))
    if not (pd.isna(f1_tpl) and pd.isna(f2_tpl) and pd.isna(f3_tpl)):
        vals = [f1_tpl if pd.notna(f1_tpl) else np.nan,
                f2_tpl if pd.notna(f2_tpl) else np.nan,
                f3_tpl if pd.notna(f3_tpl) else np.nan]
    else:
        vals = [
            df["dominant_material_fraction_1"].mean(skipna=True),
            df["dominant_material_fraction_2"].mean(skipna=True),
            df["dominant_material_fraction_3"].mean(skipna=True)
        ]
    f1_n, f2_n, f3_n = normalize_three(vals[0], vals[1], vals[2])
    df.at[idx, "dominant_material_fraction_1"] = f1_n
    df.at[idx, "dominant_material_fraction_2"] = f2_n
    df.at[idx, "dominant_material_fraction_3"] = f3_n
    
leftover = df[["dominant_material_fraction_1","dominant_material_fraction_2","dominant_material_fraction_3"]].isna().any(axis=1)
for idx in df[leftover].index:
    f1,f2,f3 = normalize_three(np.nan, np.nan, np.nan)
    df.at[idx, "dominant_material_fraction_1"] = f1
    df.at[idx, "dominant_material_fraction_2"] = f2
    df.at[idx, "dominant_material_fraction_3"] = f3

for idx in df.index:
    try:
        f1 = float(df.at[idx, "dominant_material_fraction_1"])
        f2 = float(df.at[idx, "dominant_material_fraction_2"])
        f3 = float(df.at[idx, "dominant_material_fraction_3"])
    except Exception:
        f1, f2, f3 = 1/3, 1/3, 1/3
    f1n, f2n, f3n = normalize_three(f1, f2, f3)
    df.at[idx, "dominant_material_fraction_1"] = f1n
    df.at[idx, "dominant_material_fraction_2"] = f2n
    df.at[idx, "dominant_material_fraction_3"] = f3n

for idx in df.index:
    lm = df.at[idx, "launch_mass_kg"]
    dm = df.at[idx, "dry_mass_kg"]
    if pd.notna(lm) and pd.notna(dm) and dm >= lm:
        df.at[idx, "dry_mass_kg"] = float(lm) * 0.98

frac_sum = df["dominant_material_fraction_1"].astype(float) + df["dominant_material_fraction_2"].astype(float) + df["dominant_material_fraction_3"].astype(float)
bad = ~np.isclose(frac_sum, 1.0)
for idx in df.index[bad]:
    f1 = df.at[idx, "dominant_material_fraction_1"]
    f2 = df.at[idx, "dominant_material_fraction_2"]
    f3 = df.at[idx, "dominant_material_fraction_3"]
    f1n, f2n, f3n = normalize_three(f1, f2, f3)
    df.at[idx, "dominant_material_fraction_1"] = f1n
    df.at[idx, "dominant_material_fraction_2"] = f2n
    df.at[idx, "dominant_material_fraction_3"] = f3n
    
for idx in df.index:
    m1 = str(df.at[idx, "dominant_material_1"])
    m2 = str(df.at[idx, "dominant_material_2"])
    m3 = str(df.at[idx, "dominant_material_3"])
    if m1 == m2 and m1 != "nan" and m1 != m3:
        df.at[idx, "dominant_material_2"] = m3
        df.at[idx, "dominant_material_3"] = np.nan

for col in ["dominant_material_1","dominant_material_2","dominant_material_3"]:
    if col in df.columns:
        df[col] = df[col].replace({"": np.nan})
        
df.to_excel(OUTPUT_FILE, index=False)
print("Imputation complete. Output saved to:", OUTPUT_FILE)
