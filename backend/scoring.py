# backend/scoring.py

import pandas as pd

LAUNCH_COST_PER_KG = 4000  # USD

def calculate_delta_v(orbit_altitude, inclination, eccentricity):
    altitude_factor = abs(orbit_altitude - 500) * 0.05
    inclination_factor = inclination * 2
    eccentricity_factor = eccentricity * 100
    return altitude_factor + inclination_factor + eccentricity_factor


def score_debris(debris_df, amount_required, target_orbit_altitude):
    results = []

    for _, row in debris_df.iterrows():
        predicted_mass = row["predicted_mass"]
        recovery_factor = row["recovery_factor"]

        material_match = (amount_required / predicted_mass) * 3
        material_match_score = min(material_match, 3)

        recoverable_mass = predicted_mass * recovery_factor

        if recoverable_mass > 500:
            recoverable_mass_score = 3
        elif recoverable_mass > 200:
            recoverable_mass_score = 2
        elif recoverable_mass > 50:
            recoverable_mass_score = 1
        else:
            recoverable_mass_score = 0

        delta_v = calculate_delta_v(
            row["orbit_altitude_km"],
            row["inclination_deg"],
            row["eccentricity"]
        )

        if delta_v < 150:
            dV_score = 3
            maneuver_complexity = "Low"
        elif delta_v < 300:
            dV_score = 2
            maneuver_complexity = "Medium"
        elif delta_v < 500:
            dV_score = 1
            maneuver_complexity = "High"
        else:
            dV_score = 0
            maneuver_complexity = "Very High"

        economic_gain = recoverable_mass * LAUNCH_COST_PER_KG

        if economic_gain > 2_000_000:
            economic_score = 2
        elif economic_gain > 500_000:
            economic_score = 1
        else:
            economic_score = 0

        recyclability_score = (
            material_match_score +
            recoverable_mass_score +
            dV_score +
            economic_score
        )

        results.append({
            "Debris ID": row["object_id"],
            "Orbit (km)": row["orbit_altitude_km"],
            "Delta V (m/s)": round(delta_v, 2),
            "Maneuver Complexity": maneuver_complexity,
            "Recyclability Score (0–10)": round(recyclability_score, 2)
        })

    result_df = pd.DataFrame(results)

    # Sort by recyclability score (descending)
    result_df = result_df.sort_values(
    "Recyclability Score (0–10)",
    ascending=False
    )

    return result_df
    






