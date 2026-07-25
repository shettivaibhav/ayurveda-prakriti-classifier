import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
from datetime import datetime
import random
from supabasehelper import upload_complete_analysis
from palm_color import extract_features


# =========================
# Helper function
# =========================
def generate_user_id():
    """Generate unique user ID using timestamp and random number"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_num = random.randint(1000, 9999)
    return f"{timestamp}_{random_num}"


# =========================
# Model definitions
# =========================
VIEW_ORDER = ["palm_open_front", "palm_open_back", "palm_closed_front", "palm_closed_back"]


# =========================
# Load model
# =========================
# Deployed model: patch colour (skin=palm-open back centre, palm=palm-open front
# just-above-centre) + age/sex -> RandomForest -> dominant dosha (Kapha/Pitta/Vata).
# ~67% accuracy / 46% balanced vs the practitioner's sheet. No CNN needed.
@st.cache_resource
def load_model(model_path="model_color.joblib"):
    return joblib.load(model_path)


# =========================
# Dosha display metadata
# =========================
DOSHA_DISPLAY = {
    "K": ("Kapha", "🌊"),
    "P": ("Pitta", "🔥"),
    "V": ("Vata", "🌬️"),
    "K-P": ("Kapha-Pitta", "🌊🔥"),
    "K-V": ("Kapha-Vata", "🌊🌬️"),
    "P-K": ("Pitta-Kapha", "🔥🌊"),
    "P-V": ("Pitta-Vata", "🔥🌬️"),
    "V-K": ("Vata-Kapha", "🌬️🌊"),
    "V-P": ("Vata-Pitta", "🌬️🔥"),
}


def display_name_and_emoji(class_label):
    return DOSHA_DISPLAY.get(class_label, (class_label, "⚪"))


# =========================
# Prediction: skin/palm colour patches + age/sex -> RandomForest
# =========================
def predict_dosha(images_by_angle, age, sex_m, bundle):
    """Extract the skin (back-view) and palm (front-view) colour patches + age/sex,
    run the RandomForest, and return {'K':p, 'P':p, 'V':p} dominant-dosha probs."""
    feats = extract_features(
        images_by_angle[bundle.get("skin_view", "palm_open_back")],
        images_by_angle[bundle.get("palm_view", "palm_open_front")],
        age, sex_m,
    ).reshape(1, -1)
    probs = bundle["model"].predict_proba(feats)[0]
    return {cls: float(p) for cls, p in zip(bundle["classes"], probs)}


# =========================
# Load demo images
# =========================
def load_demo_image(image_path):
    """Load demo image from file"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
    except:
        pass
    return None


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="Dosha Classifier",
    page_icon="🖐️",
    layout="centered",
)

st.title("Ayurvedic Wrist Dosha Analysis")
st.markdown(
    "Upload 4 wrist images from different angles. The model reads your skin and "
    "palm colour (with age and sex) to predict your dominant dosha once every view is uploaded."
)

# -------------------------
# Session state
# -------------------------
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

if "upload_counts" not in st.session_state:
    st.session_state.upload_counts = {v: 0 for v in VIEW_ORDER}

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "user_info" not in st.session_state:
    st.session_state.user_info = {"user_id": "", "gender": ""}

if "uploaded_to_supabase" not in st.session_state:
    st.session_state.uploaded_to_supabase = False

if "disclaimer_dismissed" not in st.session_state:
    st.session_state.disclaimer_dismissed = False

# -------------------------
# Disclaimer Dialog
# -------------------------
if not st.session_state.disclaimer_dismissed:
    with st.container(border=True):
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This is an experimental AI tool developed for educational and research purposes.

        **Important Points:**
        - This tool provides preliminary indications only and should not be considered medical advice
        - Results may not be clinically accurate
        - Consult a qualified Ayurvedic practitioner for professional dosha assessment
        - Do not make health decisions based solely on these results

        **Image Quality Guidelines:**
        - Take images in good lighting conditions
        - Use a clear, light-colored background
        - Ensure wrist is clearly visible and in focus

        **Data Usage Consent:**
        - Your uploaded wrist images and gender information will be stored securely
        - Data may be used for educational and research purposes to improve the model
        - All data is anonymized and handled in accordance with privacy standards

        **Your Privacy:**
        - Images are compressed and stored securely in cloud storage
        - Only anonymized User IDs are used for data tracking
        """)

        if st.button("Close", type="primary", use_container_width=True):
            st.session_state.disclaimer_dismissed = True
            st.rerun()

# -------------------------
# Load model
# -------------------------
try:
    model_bundle = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'model_color.joblib' not found.")
    st.stop()

# -------------------------
# User Information Collection
# -------------------------
st.markdown("### User Information")
gender = st.selectbox(
    "Select Gender",
    options=["", "male", "female"],
    index=0 if not st.session_state.user_info["gender"] else
          (["male", "female"].index(st.session_state.user_info["gender"]) + 1),
    help="Required for analysis"
)

if gender:
    st.session_state.user_info["gender"] = gender
    # Auto-generate user_id if not already set
    if not st.session_state.user_info["user_id"]:
        st.session_state.user_info["user_id"] = generate_user_id()
else:
    st.warning("⚠️ Please select gender before uploading images")
    st.stop()

age = st.number_input(
    "Age", min_value=1, max_value=120,
    value=int(st.session_state.user_info.get("age") or 25), step=1,
    help="Used by the model along with the hand images",
)
st.session_state.user_info["age"] = int(age)

st.markdown("---")

# -------------------------
# Wrist View Definitions with Demo Images
# -------------------------
ANGLE_CONFIG = {
    "palm_open_front": {
        "title": "Palm Open – Front View",
        "description": "Hand open, palm facing camera",
        "demo_image": "demo_images/wristfrontopen.jpg",
        "status": "pending"
    },
    "palm_open_back": {
        "title": "Palm Open – Back View",
        "description": "Hand open, back of hand facing camera",
        "demo_image": "demo_images/wristbackopen.jpg",
        "status": "pending"
    },
    "palm_closed_front": {
        "title": "Palm Closed – Front View",
        "description": "Hand closed as fist, knuckles facing camera",
        "demo_image": "demo_images/wristfrontclose.jpg",
        "status": "pending"
    },
    "palm_closed_back": {
        "title": "Palm Closed – Back View",
        "description": "Hand closed as fist, back of hand facing camera",
        "demo_image": "demo_images/wristbackclose.png",
        "status": "pending"
    }
}

# Update status based on what's uploaded
for angle in ANGLE_CONFIG:
    if angle in st.session_state.uploaded_images:
        ANGLE_CONFIG[angle]["status"] = "uploaded"

# -------------------------
# REQUIRED WRIST VIEWS - 2x2 Grid with Demo Images
# -------------------------
st.markdown("### 📋 Required Wrist Views")
st.markdown("Upload images for all 4 positions below. Click to expand each section:")

# Create 2x2 grid
grid_col1, grid_col2 = st.columns(2)

# Function to create view section
def create_view_section(angle_key, col):
    config = ANGLE_CONFIG[angle_key]

    # Determine expander state and title
    is_uploaded = config["status"] == "uploaded"
    expander_title = f"✅ {config['title']}" if is_uploaded else f"⬜ {config['title']}"

    with col:
        with st.expander(expander_title, expanded=not is_uploaded):
            if is_uploaded:
                # Show uploaded status
                st.success("**Uploaded** ✓")
                st.image(st.session_state.uploaded_images[angle_key],
                        caption="Your uploaded image", width=150)

                # Option to re-upload
                if st.button(f"Replace Image", key=f"replace_{angle_key}"):
                    del st.session_state.uploaded_images[angle_key]
                    st.session_state.prediction = None
                    st.rerun()

            else:
                # Show demo image and upload interface
                demo_img = load_demo_image(config["demo_image"])
                if demo_img:
                    st.image(demo_img, caption="Example of this view", width=150)

                st.caption(config["description"])

                # Upload interface
                uploaded_file = st.file_uploader(
                    "Upload your wrist image",
                    type=["jpg", "jpeg", "png"],
                    key=f"upload_{angle_key}_{st.session_state.upload_counts[angle_key]}",
                    label_visibility="collapsed"
                )

                if uploaded_file:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.session_state.uploaded_images[angle_key] = image
                    st.session_state.upload_counts[angle_key] += 1
                    st.session_state.prediction = None
                    st.rerun()

# Row 1 - Palm Open Views
create_view_section("palm_open_front", grid_col1)
create_view_section("palm_open_back", grid_col2)

# Row 2 - Palm Closed Views
grid_col3, grid_col4 = st.columns(2)
create_view_section("palm_closed_front", grid_col3)
create_view_section("palm_closed_back", grid_col4)

# Progress indicator
uploaded_count = len(st.session_state.uploaded_images)
if uploaded_count < 4:
    st.progress(uploaded_count / 4, text=f"Progress: {uploaded_count}/4 views uploaded")
    st.caption(f"*{4 - uploaded_count} more view(s) needed before analysis can run*")
else:
    st.success("✅ All 4 views uploaded!")

st.markdown("---")

# -------------------------
# Run colour-based prediction once all 4 views are present
# -------------------------
if uploaded_count == 4 and st.session_state.prediction is None:
    with st.spinner("Analyzing skin & palm colour..."):
        sex_m = 1 if st.session_state.user_info["gender"] == "male" else 0
        st.session_state.prediction = predict_dosha(
            st.session_state.uploaded_images,
            st.session_state.user_info["age"], sex_m, model_bundle,
        )

# -------------------------
# FINAL DOSHA PREDICTION
# -------------------------
if st.session_state.prediction:
    st.markdown("### 🎯 **Dosha Analysis**")

    probs = st.session_state.prediction
    top_class = max(probs.items(), key=lambda x: x[1])[0]
    confidence = probs[top_class]
    dosha_name, emoji = display_name_and_emoji(top_class)

    # Display final result
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown(f"# {emoji}")
        with cols[1]:
            st.markdown(f"## {dosha_name}")
            st.caption(f"Confidence: {confidence * 100:.1f}% · From skin & palm colour + age/sex")

    # Soft secondary tendency (the 2nd-highest dosha) — NOT a validated dual-dosha claim
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) > 1 and ranked[1][1] >= 0.25:
        sec_name, sec_emoji = display_name_and_emoji(ranked[1][0])
        st.caption(f"Secondary tendency: {sec_name} {sec_emoji} "
                   f"({ranked[1][1]*100:.0f}%) — lower confidence, not a clinical dual-dosha result")

    # All-class probabilities
    st.markdown("##### Full Probability Distribution:")
    for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        name, emoji = display_name_and_emoji(cls)
        col1, col2, col3 = st.columns([2, 4, 2])
        with col1:
            st.markdown(f"**{name}** {emoji}")
        with col2:
            st.progress(p)
        with col3:
            st.markdown(f"**{p*100:.1f}%**")

    # Auto-save to cloud if not already uploaded
    if not st.session_state.uploaded_to_supabase:
        if st.session_state.user_info["user_id"] and st.session_state.user_info["gender"]:
            with st.spinner("💾 Saving to cloud..."):
                user_id = st.session_state.user_info["user_id"]
                gender = st.session_state.user_info["gender"]
                images_dict = st.session_state.uploaded_images

                # The 3-class model outputs real Vata/Pitta/Kapha probabilities,
                # so the Supabase vata_prob/pitta_prob/kapha_prob columns are
                # populated directly. predicted_dosha stores the dominant dosha.
                dosha_probs = {
                    "Vata": probs.get("V", 0.0),
                    "Pitta": probs.get("P", 0.0),
                    "Kapha": probs.get("K", 0.0),
                }

                result = upload_complete_analysis(
                    user_id=user_id,
                    gender=gender,
                    images_dict=images_dict,
                    probabilities=dosha_probs,
                    final_dosha=f"{dosha_name} ({top_class})",
                    confidence=confidence
                )

                if result["success"]:
                    st.session_state.uploaded_to_supabase = True
                    st.session_state.upload_record_id = result["record_id"]
                    st.rerun()
                else:
                    st.error(f"❌ Cloud save failed: {', '.join(result['errors'])}")

    # Show success messages
    st.balloons()
    st.success("✅ Analysis complete - dominant dosha from skin & palm colour!")

    if st.session_state.uploaded_to_supabase:
        st.success(f"☁️ Saved to cloud! User ID: **{st.session_state.user_info['user_id']}**")

        # Show reset button only after successful upload
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            st.session_state.uploaded_images.clear()
            st.session_state.upload_counts = {v: 0 for v in VIEW_ORDER}
            st.session_state.prediction = None
            st.session_state.uploaded_to_supabase = False
            st.session_state.user_info = {"user_id": "", "gender": ""}
            st.rerun()

# -------------------------
# INFO SECTION
# -------------------------
with st.expander("📚 Understanding Your Dosha Results", expanded=False):
    st.markdown("""
    ### What Are Doshas?

    In Ayurveda, doshas are the three fundamental energies that govern our physical and mental processes.

    **🌬️ Vata (Air + Ether)**
    - *Qualities:* Dry, light, cold, rough, subtle, mobile
    - *Governs:* Movement, breathing, circulation, nerve impulses
    - *When balanced:* Creative, energetic, adaptable
    - *When imbalanced:* Anxiety, dry skin, constipation, insomnia

    **🔥 Pitta (Fire + Water)**
    - *Qualities:* Hot, sharp, light, liquid, oily
    - *Governs:* Digestion, metabolism, body temperature
    - *When balanced:* Intelligent, focused, good leadership
    - *When imbalanced:* Anger, inflammation, heartburn, skin rashes

    **🌊 Kapha (Earth + Water)**
    - *Qualities:* Heavy, slow, cool, oily, smooth, dense
    - *Governs:* Structure, stability, lubrication
    - *When balanced:* Calm, loving, strong, grounded
    - *When imbalanced:* Weight gain, lethargy, congestion, depression

    ### Understanding Your Results

    - **Single-Dosha:** One energy is predominant
    - **Dual-Dosha:** Two energies are jointly predominant (most common)

    *Note: This analysis is based on wrist characteristics and provides an indication of your predominant dosha(s), predicted by averaging all 4 uploaded views.*
    """)

st.markdown("---")
st.markdown("*Ayurvedic analysis using AI • Dominant dosha from skin & palm colour + age/sex*")
