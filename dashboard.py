import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# ─── CONFIG ───────────────────────────────────────
app_URL = "http://localhost:8000"
# ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Parcel Damage Detection",
    page_icon="📦",
    layout="wide"
)

# ─── HEADER ───────────────────────────────────────
st.title("📦 Parcel Damage Detection System")
st.markdown("**AI-powered logistics damage detection using EfficientNet-B5**")
st.divider()

# ─── SIDEBAR ──────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    try:
        metrics = requests.get(f"{app_URL}/metrics").json()
        st.metric("Overall Accuracy", metrics["overall_acc"])
        st.metric("Damaged Accuracy", metrics["damaged_acc"])
        st.metric("Intact Accuracy",  metrics["intact_acc"])
        st.metric("Training Images",  metrics["augmented"])
        st.metric("Model",            metrics["model"])
        st.success("🟢 app Connected")
    except:
        st.error("🔴 app Offline — run app.py first")

# ─── MAIN ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Box Image")
    uploaded = st.file_uploader(
        "Choose a box photo",
        type=["jpg", "jpeg", "png"],
        help="Upload a parcel/box image to check for damage"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔍 Prediction Result")

    if uploaded:
        with st.spinner("Analyzing image..."):
            try:
                # Send to app
                files    = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(f"{app_URL}/predict", files=files)
                result   = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]
                status     = result["status"]
                dmg_prob   = float(result["damaged_prob"].replace("%", ""))
                int_prob   = float(result["intact_prob"].replace("%", ""))

                # Show result
                if prediction == "damaged":
                    st.error(f"## ❌ DAMAGED")
                    st.error(f"### Status: {status}")
                else:
                    st.success(f"## ✅ INTACT")
                    st.success(f"### Status: {status}")

                st.metric("Confidence", confidence)

                # Probability bars
                st.markdown("**Probability Breakdown:**")
                st.progress(dmg_prob / 100, text=f"Damaged: {dmg_prob:.1f}%")
                st.progress(int_prob / 100, text=f"Intact:  {int_prob:.1f}%")

                # Store in session history
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "File"      : result["filename"],
                    "Prediction": prediction.upper(),
                    "Confidence": confidence,
                    "Status"    : "REJECT" if prediction == "damaged" else "ACCEPT"
                })

            except Exception as e:
                st.error(f"app Error: {e}")
                st.info("Make sure app.py is running!")
    else:
        st.info("👈 Upload an image to get started")

# ─── HISTORY ──────────────────────────────────────
st.divider()
st.subheader("📋 Prediction History")

if "history" in st.session_state and st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    # Stats
    total    = len(df)
    damaged  = len(df[df["Prediction"] == "DAMAGED"])
    intact   = len(df[df["Prediction"] == "INTACT"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scanned", total)
    c2.metric("Damaged",       damaged)
    c3.metric("Intact",        intact)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Upload images above.")