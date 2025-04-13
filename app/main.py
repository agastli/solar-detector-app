print("✅ Streamlit app starting...")

import torch
torch.classes.__path__ = []  # Fix for Streamlit Cloud with torch bindings

import sys
import os
import requests
import streamlit as st
from PIL import Image
import tempfile

# Fix module resolution and file watcher on Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Attempt to import modules
try:
    from src.utils.logger import logger
    from src.detection.infer import detect_panels
    from src.utils.energy import estimate_energy
    from src.utils.image_utils import get_image_dimensions
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    raise

# ------------------ Page Config ------------------ #
st.set_page_config(page_title="Solar Panel Detection", layout="wide")
st.write("✅ Streamlit UI loaded")

# ------------------ Title ------------------ #
st.title("☀️ Solar Panel Detection and Energy Estimation")
st.markdown("Upload satellite/aerial imagery and get solar panel detection with estimated energy yield.")

# ------------------ File Upload ------------------ #
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "tif", "tiff"])

# ------------------ Sidebar ------------------ #
st.sidebar.header("⚙️ Model & Location Settings")

# Model selection
model_options = {
    "Model3": "models/best3.pt",
    "Model2": "models/best2.pt",
    "Model1": "models/best1.pt"
}
selected_model_name = st.sidebar.selectbox("YOLOv8 Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# Location input
latitude = st.sidebar.number_input("Latitude", value=36.45028, format="%.5f")
longitude = st.sidebar.number_input("Longitude", value=10.73389, format="%.5f")

# Panel efficiency and losses
panel_efficiency = st.sidebar.number_input("Panel efficiency (%)", value=18.5, format="%.1f") / 100
system_loss = st.sidebar.number_input("System losses (%)", value=10.0, format="%.1f") / 100

# Scale conversion
scale_m_per_px = st.sidebar.number_input(
    "Pixel-to-Meter Scale (m/px)", min_value=0.01, max_value=10.0,
    value=0.08, step=0.001, format="%.3f"
)

# ------------------ PVGIS API ------------------ #
def get_avg_irradiance(lat, lon):
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&outputformat=json&peakpower=1&loss=14"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data['outputs']['totals']['fixed']['E_y'] / 365, url, None
        return None, url, f"PVGIS error: status {res.status_code}"
    except Exception as e:
        return None, url, str(e)

irradiance_kwh_per_m2_day, pvgis_url, error_msg = get_avg_irradiance(latitude, longitude)

if irradiance_kwh_per_m2_day:
    st.sidebar.success(f"✔️ Irradiance: {irradiance_kwh_per_m2_day:.2f} kWh/m²/day (PVGIS)")
else:
    st.sidebar.warning("⚠️ Could not fetch irradiance automatically.")
    irradiance_kwh_per_m2_day = st.sidebar.number_input("Manual Irradiance (kWh/m²/day)", value=5.5, format="%.2f")
    if st.sidebar.button("🔄 Retry PVGIS"):
        st.experimental_rerun()
    with st.sidebar.expander("🔍 PVGIS Debug Info"):
        st.code(pvgis_url, language="text")
        st.text(error_msg)

# ------------------ Detection & Estimation ------------------ #
if not uploaded_file:
    st.info("📂 Please upload an image to begin.")
    st.stop()

logger.info("📥 Image uploaded")
st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded_file.read())
    image_path = tmp.name

# Get dimensions
width, height = get_image_dimensions(image_path)
st.markdown(f"**Image Dimensions:** `{width}px × {height}px`")

# Run detection
with st.spinner("🔍 Detecting solar panels..."):
    detected_img, detection_df = detect_panels(image_path, selected_model_path)

if detection_df is not None and not detection_df.empty:
    detection_df["area_px"] = (detection_df["xmax"] - detection_df["xmin"]) * (detection_df["ymax"] - detection_df["ymin"])
    detection_df["area_m2"] = detection_df["area_px"] * (scale_m_per_px ** 2)
    total_area_px = detection_df["area_px"].sum()
    total_area_m2 = detection_df["area_m2"].sum()

    st.subheader("✅ Detection Results")
    st.image(detected_img, caption="Detected Panels", use_container_width=True)
    st.dataframe(detection_df[["name", "confidence", "area_px", "area_m2"]])
    st.markdown(f"**Total Area (px²):** `{total_area_px:.2f}`")
    st.markdown(f"**Estimated Area (m²):** `{total_area_m2:.2f}`")

    # Energy estimation
    daily_energy_kwh, yearly_energy_kwh = estimate_energy(
        total_area_m2, irradiance_kwh_per_m2_day, panel_efficiency, system_loss
    )
    st.markdown(f"**Estimated Daily Output:** `{daily_energy_kwh:.2f} kWh`")
    st.markdown(f"**Estimated Yearly Output:** `{yearly_energy_kwh:.2f} kWh`")
else:
    st.warning("🚫 No solar panels detected. Try a different image.")
    logger.warning("Detection result: empty")
