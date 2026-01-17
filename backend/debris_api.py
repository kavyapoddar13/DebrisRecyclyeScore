from fastapi import APIRouter
import pandas as pd
from backend.scoring import score_debris

router = APIRouter()

# --------------------------------------------------
# MATERIAL NORMALIZATION
# --------------------------------------------------
def normalize_material(material: str) -> str:
    material = material.lower()

    if "alum" in material:
        return "aluminium"
    if "barium" in material:
        return "barium"
    if "titan" in material:
        return "titanium"
    if "steel" in material:
        return "steel"
    if "composite" in material:
        return "composite"
    if "carbon" in material:
        return "carbon"
    if "copper" in material:
        return "copper"

    return "other"


@router.post("/score")
def score_debris_api(
    amount_required: float,
    target_orbit_altitude: float,
    material_needed: str
):
    # --------------------------------------------------
    # LOAD ML-GENERATED DATASET
    # --------------------------------------------------
    debris_df = pd.read_excel(
        "ml_implementation/norad_mass_dataset_500_imputed.xlsx"
    )

    # --------------------------------------------------
    # DERIVED / NORMALIZED COLUMNS
    # --------------------------------------------------

    # 1️⃣ Predicted mass (best proxy)
    debris_df["predicted_mass"] = debris_df["dry_mass_kg"]

    # 2️⃣ Recovery factor (material dominance)
    debris_df["recovery_factor"] = (
        debris_df["dominant_material_fraction_1"]
        .fillna(0)
        .clip(0, 1)
    )

    # 3️⃣ Mean orbit altitude
    debris_df["orbit_altitude_km"] = (
        debris_df["perigee_km"] + debris_df["apogee_km"]
    ) / 2

    # 4️⃣ Orbital eccentricity
    debris_df["eccentricity"] = (
        debris_df["apogee_km"] - debris_df["perigee_km"]
    ) / (
        debris_df["apogee_km"] + debris_df["perigee_km"]
    )
    debris_df["eccentricity"] = debris_df["eccentricity"].fillna(0)

    # --------------------------------------------------
    # MATERIAL NORMALIZATION & FILTERING
    # --------------------------------------------------
    for col in [
        "dominant_material_1",
        "dominant_material_2",
        "dominant_material_3"
    ]:
        debris_df[col + "_norm"] = (
            debris_df[col]
            .fillna("")
            .apply(normalize_material)
        )

    material_needed_norm = normalize_material(material_needed)

    debris_df = debris_df[
        (debris_df["dominant_material_1_norm"] == material_needed_norm) |
        (debris_df["dominant_material_2_norm"] == material_needed_norm) |
        (debris_df["dominant_material_3_norm"] == material_needed_norm)
    ]

    # --------------------------------------------------
    # CALL SCORING ENGINE
    # --------------------------------------------------
    result_df = score_debris(
        debris_df=debris_df,
        amount_required=amount_required,
        target_orbit_altitude=target_orbit_altitude
    )

    # --------------------------------------------------
    # SORT BY RECYCLABILITY SCORE (DESCENDING)
    # --------------------------------------------------
    result_df = result_df.sort_values(
        "Recyclability Score (0–10)",
        ascending=False
    )

    # --------------------------------------------------
    # RETURN JSON
    # --------------------------------------------------
    return result_df.to_dict(orient="records")
