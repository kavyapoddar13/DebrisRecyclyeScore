import streamlit as st
import smtplib
import random
import requests
import pandas as pd
from email.mime.text import MIMEText

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Space Debris Recycling Platform", layout="centered")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f3d2b 0%, #2b1d14 100%) !important;
}

.block-container {
    max-width: 500px !important;
    padding-top: 3rem !important;
}

.header-container {
    background: #f5f1e8;
    padding: 1.2rem;
    border-radius: 15px 15px 0 0;
    text-align: center;
    box-shadow: 0 6px 15px rgba(0,0,0,0.25);
}

.content-card {
    background: #f5f1e8;
    padding: 1.5rem;
    border-radius: 0 0 15px 15px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
}

.stButton > button {
    background: linear-gradient(135deg, #c9a24d 0%, #4a6b3c 100%) !important;
    color: #2b1d14 !important;
    border-radius: 10px !important;
    width: 100% !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# OTP FUNCTIONS
# --------------------------------------------------
def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(receiver_email, otp):
    sender_email = "kavyapoddar13@gmail.com"
    sender_app_password = "kmoh nlre lxzy ogap"

    msg = MIMEText(f"Your OTP is: {otp}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

# --------------------------------------------------
# MATERIAL NORMALIZATION (FRONTEND)
# --------------------------------------------------
def normalize_material(material: str):
    material = material.lower()
    if "alum" in material:
        return "Aluminium"
    if "barium" in material:
        return "Barium"
    if "titan" in material:
        return "Titanium"
    if "steel" in material:
        return "Steel"
    if "composite" in material:
        return "Composite"
    if "carbon" in material:
        return "Carbon"
    if "copper" in material:
        return "Copper"
    if "nickel" in material:
        return "Nickel"
    if "iron" in material:
        return "Iron"
    return None

@st.cache_data
def load_material_options():
    df = pd.read_excel("ml_implementation/norad_mass_dataset_500_imputed.xlsx")
    materials = set()

    for col in [
        "dominant_material_1",
        "dominant_material_2",
        "dominant_material_3"
    ]:
        for val in df[col].dropna():
            norm = normalize_material(val)
            if norm:
                materials.add(norm)

    return sorted(list(materials))

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "otp" not in st.session_state:
    st.session_state.otp = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "results_df" not in st.session_state:
    st.session_state.results_df = None

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="header-container">
    <div style="font-size:2rem;">🔐</div>
    <h2 style="color:#667eea;">Space Debris Recycling Platform</h2>
    <p style="color:#6c757d;">Secure Access</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content-card">', unsafe_allow_html=True)

# --------------------------------------------------
# AUTH UI
# --------------------------------------------------
if not st.session_state.authenticated:

    st.markdown("### 📧 Email Verification")
    email = st.text_input("Email", placeholder="example@email.com", label_visibility="collapsed")

    if st.button("Send OTP"):
        if not email:
            st.error("Please enter your email")
        else:
            otp = generate_otp()
            st.session_state.otp = otp
            send_otp_email(email, otp)
            st.success("OTP sent successfully!")

    entered_otp = st.text_input("Enter OTP", placeholder="000000", max_chars=6)

    if st.button("Verify OTP"):
        if entered_otp == st.session_state.otp:
            st.session_state.authenticated = True
            st.rerun()   # ✅ UPDATED (no warning)
        else:
            st.error("Invalid OTP")

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
else:
    st.success("Logged in successfully")

    st.markdown("### 🚀 Space Agency Requirements")

    material_needed = st.selectbox(
        "Required Material",
        load_material_options()
    )

    amount_needed = st.number_input("Material Required (kg)", min_value=1.0)
    orbit_altitude = st.number_input(
        "Target Orbit Altitude (km)",
        min_value=200.0,
        max_value=2000.0
    )

    if st.button("Find Suitable Debris"):
        API_URL = "http://localhost:8000/debris/score"
        response = requests.post(
            API_URL,
            params={
                "amount_required": amount_needed,
                "target_orbit_altitude": orbit_altitude,
                "material_needed": material_needed
            }
        )

        if response.status_code == 200:
            st.session_state.results_df = pd.DataFrame(response.json())
        else:
            st.error("Backend error")

    # ---------------- DISPLAY RESULTS ----------------
    if st.session_state.results_df is not None:

        df = st.session_state.results_df.copy()

        sort_option = st.selectbox(
            "Sort by Maneuver Complexity",
            ["Default (Recyclability Score)", "Low → Very High", "Very High → Low"]
        )

        if sort_option != "Default (Recyclability Score)":
            rank = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
            df["rank"] = df["Maneuver Complexity"].map(rank)
            df = df.sort_values(
                "rank",
                ascending=(sort_option == "Low → Very High")
            )
            df = df.drop(columns="rank")

        st.dataframe(df)

st.markdown("</div>", unsafe_allow_html=True)
