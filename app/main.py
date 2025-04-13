print("✅ Streamlit app starting...")

import torch
torch.classes.__path__ = []  # Fix for Streamlit Cloud with torch bindings

import sys
import os
import requests
import streamlit as st
from PIL import Image
import tempfile

# Fix module resolution and file watcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ------------------ Safe Module Imports ------------------ #
try:
    from src.utils.logger import logger
    from src.detection.infer import detect_panels
    from src.utils.energy import estimate_energy
    from src.utils.image_utils import get_image_dimensions
except Exception as e:
    st.error(f"❌ Failed to import modules: {e}")
    raise

# ------------------ Streamlit Page Config ------------------ #
st.set_page_config(page_title="Solar Panel Detection", layout="wide")
st.title("☀️ Solar Panel Detection and Energy Estimation")
st.markdown("Upload a satellite or aerial image to detect solar panels and estimate energy yield.")

# ------------------ Sidebar Configuration ------------------ #
st.sidebar.header("Model & Location Settings")
model_options = {
    "Model3": "models/best3.pt",
    "Model2": "models/best2.pt",
    "Model1": "models/best1.pt"
}
selected_model = st.sidebar.selectbox("Choose Detection Model", list(model_options.keys()))
model_path = model_options[selected_model]

latitude = st.sidebar.number_input("Latitude", value=36.45028, format="%.5f")
longitude = st.sidebar.number_input("Longitude", value=10.73389, format="%.5f")
panel_efficiency = st.sidebar.number_input("Panel efficiency (%)", value=18.5, format="%.1f") / 100
system_loss = st.sidebar.number_input("System loss (%)", value=10.0, format="%.1f") / 100
scale_m_per_px = st.sidebar.number_input("Pixel-to-Meter Scale (m/px)", value=0.08, format="%.3f", step=0.001)

# ------------------ PVGIS Irradiance Fetch ------------------ #
def get_avg_irradiance(lat, lon):
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&outputformat=json&peakpower=1&loss=14"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data['outputs']['totals']['fixed']['E_y'] / 365, url, None
        return None, url, f"PVGIS responded with status code {res.status_code}"
    except Exception as e:
        return None, url, f"Error fetching irradiance: {str(e)}"

irradiance, api_url, error_msg = get_avg_irradiance(latitude, longitude)

if irradiance:
    st.sidebar.success(f"☀️ Avg Irradiance: {irradiance:.2f} kWh/m²/day")
else:
    st.sidebar.warning("⚠️ Failed to fetch irradiance from PVGIS. Please enter it manually.")
    irradiance = st.sidebar.number_input("Manual Irradiance", value=5.5, format="%.2f")
    if st.sidebar.button("🔄 Retry PVGIS"):
        st.experimental_rerun()
    with st.sidebar.expander("🔍 PVGIS Debug Info"):
        st.code(api_url)
        st.code(error_msg)

# ------------------ Image Upload and Detection ------------------ #
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("📂 Please upload a valid image to proceed.")
    st.stop()

# Show uploaded image
st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
logger.info("📥 Image uploaded successfully.")

# Save temp image
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded_file.read())
    image_path = tmp.name

# Get image dimensions
width, height = get_image_dimensions(image_path)
st.markdown(f"🖼 **Image Dimensions:** `{width}px × {height}px`")

# Run detection
with st.spinner("🔍 Running detection..."):
    detected_img, detection_df = detect_panels(image_path, model_path)

# Show detection results
if detection_df is not None and not detection_df.empty:
    detection_df["area_px"] = (detection_df["xmax"] - detection_df["xmin"]) * (detection_df["ymax"] - detection_df["ymin"])
    detection_df["area_m2"] = detection_df["area_px"] * (scale_m_per_px ** 2)
    total_area_px = detection_df["area_px"].sum()
    total_area_m2 = detection_df["area_m2"].sum()

    st.subheader("✅ Detected Panels")
    st.image(detected_img, caption="Detected Panels with Bounding Boxes", use_container_width=True)
    st.dataframe(detection_df[["name", "confidence", "area_px", "area_m2"]])
    st.markdown(f"📐 **Total Area (px²):** `{total_area_px:.2f}`")
    st.markdown(f"📏 **Estimated Area (m²):** `{total_area_m2:.2f}`")

    # Estimate energy output
    daily_kwh, yearly_kwh = estimate_energy(total_area_m2, irradiance, panel_efficiency, system_loss)
    st.markdown(f"⚡ **Estimated Daily Energy Output:** `{daily_kwh:.2f} kWh`")
    st.markdown(f"📅 **Estimated Yearly Energy Output:** `{yearly_kwh:.2f} kWh`")
else:
    st.warning("🚫 No panels detected. Try uploading a different image.")
    logger.warning("Detection dataframe is empty.")
