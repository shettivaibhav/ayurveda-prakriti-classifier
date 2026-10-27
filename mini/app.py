# Trigger hot reload: pipeline fix applied
import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
from datetime import datetime
import random
from pathlib import Path
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
@st.cache_resource
def load_model(model_name="unified_prakriti_pipeline.joblib"):
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / model_name
    if not model_path.exists():
        model_path = base_dir / "model_color.joblib"
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
# Prediction: 4-View Unified CV + ML Pipeline
# =========================
def predict_dosha(images_by_angle, age, sex_m, bundle):
    """Extract features across all 4 anatomical views + age/sex and predict dominant dosha."""
    if "pipeline" in bundle:
        from unified_cv import extract_participant_features
        feats = extract_participant_features(images_by_angle, age, sex_m).reshape(1, -1)
        pipeline = bundle["pipeline"]
        probs = pipeline.predict_proba(feats)[0]
        return {cls: float(p) for cls, p in zip(bundle["classes"], probs)}
    else:
        # Fallback for legacy 22-feature bundle
        feats = extract_features(
            images_by_angle[bundle.get("skin_view", "palm_open_back")],
            images_by_angle[bundle.get("palm_view", "palm_open_front")],
            age, sex_m,
        ).reshape(1, -1)
        probs = bundle["model"].predict_proba(feats)[0]
        return {cls: float(p) for cls, p in zip(bundle["classes"], probs)}


# =========================
# Clinical Modulation (Doctor-in-the-Loop)
# =========================
def apply_clinical_modulation(raw_probs, vein="Auto (from images)", mole="Auto (from images)", skin="Auto (from images)"):
    """
    Apply classical Ayurvedic likelihood modulation to CV-extracted probabilities.
    When set to 'Auto (from images)', likelihood multiplier is 1.0 (pure CV).
    When clinical markers are specified by practitioner, modulates V/P/K and renormalizes.
    """
    multipliers = {"V": 1.0, "P": 1.0, "K": 1.0}

    # 1. Vein Visibility
    if "High" in vein:  # Prominent / Superficial (Vata hallmark)
        multipliers["V"] *= 1.45
        multipliers["K"] *= 0.70
    elif "Low" in vein:   # Deep / Concealed (Kapha hallmark)
        multipliers["K"] *= 1.45
        multipliers["V"] *= 0.65
    elif "Medium" in vein:
        multipliers["P"] *= 1.15

    # 2. Mole on Hand / Wrist
    if mole == "Yes":
        multipliers["P"] *= 1.30
        multipliers["V"] *= 1.15
        multipliers["K"] *= 0.85
    elif mole == "No":
        multipliers["K"] *= 1.10
        multipliers["P"] *= 0.95

    # 3. Skin Type / Texture
    if "Dry" in skin:  # Ruksha (Vata hallmark)
        multipliers["V"] *= 1.60
        multipliers["K"] *= 0.60
        multipliers["P"] *= 0.90
    elif "Rough" in skin:  # Khara (Vata hallmark)
        multipliers["V"] *= 1.50
        multipliers["K"] *= 0.65
        multipliers["P"] *= 0.90
    elif "Smooth" in skin or "Normal" in skin:  # Snigdha / Slakshna (Kapha/Pitta)
        multipliers["K"] *= 1.25
        multipliers["P"] *= 1.10
        multipliers["V"] *= 0.80
    elif "Oily" in skin:  # Sneha (Kapha / Pitta)
        multipliers["K"] *= 1.35
        multipliers["P"] *= 1.30
        multipliers["V"] *= 0.60

    # Calculate modulated unnormalized values
    modulated = {cls: raw_probs.get(cls, 0.0) * multipliers.get(cls, 1.0) for cls in raw_probs}
    total = sum(modulated.values())
    if total > 0:
        return {cls: val / total for cls, val in modulated.items()}
    return raw_probs


# =========================
# Load demo images
# =========================
def load_demo_image(image_path):
    """Load demo image from file"""
    try:
        base_dir = Path(__file__).resolve().parent
        resolved_path = base_dir / image_path
        if resolved_path.exists():
            return Image.open(resolved_path)
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

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Wrist Dosha Analysis", "WhatsApp Batch Import", "🩺 Doctor Clinical Portal"])

if page == "WhatsApp Batch Import":
    from whatsapp_ui import render_whatsapp_batch_page
    render_whatsapp_batch_page()
    st.stop()

if page == "🩺 Doctor Clinical Portal":
    from doctor_ui import render_doctor_portal_page
    render_doctor_portal_page()
    st.stop()

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

if "raw_cv_prediction" not in st.session_state:
    st.session_state.raw_cv_prediction = None

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

# -------------------------
# Optional Clinical Observations (Doctor-in-the-Loop)
# -------------------------
with st.expander("🩺 Optional Clinical Observations (Doctor-in-the-Loop)", expanded=False):
    st.caption(
        "By default, the classifier automatically relies on the Computer Vision pipeline from your 4 hand images. "
        "If you or an Ayurvedic practitioner specify any of these clinical markers, the model incorporates them "
        "using classical Ayurvedic diagnostic correlations."
    )
    clin_col1, clin_col2, clin_col3 = st.columns(3)
    with clin_col1:
        vein_visibility = st.selectbox(
            "Vein Visibility (Sira)",
            options=["Auto (from images)", "Low (Deep / Concealed)", "Medium", "High (Prominent / Superficial)"],
            index=0,
            help="High prominence indicates Vata; concealed/deep indicates Kapha.",
            key="clin_veins"
        )
        skin_color_in = st.selectbox(
            "Skin Color (Varna)",
            options=["Auto (from images)", "Fair", "Medium", "Dark"],
            index=0,
            help="Skin tone on dorsal hand/wrist.",
            key="clin_sc"
        )
    with clin_col2:
        mole_visibility = st.selectbox(
            "Mole on Hand / Wrist (Tilaka)",
            options=["Auto (from images)", "No", "Yes"],
            index=0,
            help="Presence of moles/freckles typically associates with Pitta/Vata.",
            key="clin_moles"
        )
        palm_color_in = st.selectbox(
            "Palm Color (Kara Tala)",
            options=["Auto (from images)", "Pink-Red", "Pink-White", "Brown"],
            index=0,
            help="Erythema and tone of inner palm.",
            key="clin_pc"
        )
    with clin_col3:
        skin_type = st.selectbox(
            "Skin Texture / Type (Sparsha)",
            options=["Auto (from images)", "Normal / Smooth", "Dry", "Rough", "Oily"],
            index=0,
            help="Dry/Rough skin strongly correlates with Vata; Smooth/Oily with Kapha or Pitta.",
            key="clin_skin"
        )
        hair_thickness_in = st.selectbox(
            "Skin Hair Thickness (Kesha)",
            options=["Auto (from images)", "Thin / Sparse", "Medium", "Thick"],
            index=0,
            help="Forearm/wrist hair density.",
            key="clin_hair"
        )

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
                uploaded_img = st.session_state.uploaded_images[angle_key]
                st.success("**Uploaded** ✓")
                st.image(uploaded_img, caption="Your uploaded image", width=150)

                # Image quality validation check
                from unified_cv import validate_image_quality
                q_res = validate_image_quality(uploaded_img)
                if not q_res["valid"]:
                    st.warning("⚠️ " + "; ".join(q_res["issues"]))
                else:
                    st.caption(f"✓ Quality verified (Sharpness: {q_res['blur_score']:.0f})")

                # Option to re-upload
                if st.button(f"Replace Image", key=f"replace_{angle_key}"):
                    del st.session_state.uploaded_images[angle_key]
                    st.session_state.raw_cv_prediction = None
                    st.session_state.prediction = None
                    st.session_state.uploaded_to_supabase = False
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
                    st.session_state.raw_cv_prediction = None
                    st.session_state.prediction = None
                    st.session_state.uploaded_to_supabase = False
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
if uploaded_count == 4 and st.session_state.raw_cv_prediction is None:
    with st.spinner("Analyzing anatomical features & extracting 7 clinical markers..."):
        sex_m = 1 if st.session_state.user_info["gender"] == "male" else 0
        from unified_cv import extract_cv_clinical_markers
        
        # 1. Unified CV feature extraction
        st.session_state.cv_clinical_markers = extract_cv_clinical_markers(st.session_state.uploaded_images)
        st.session_state.raw_cv_prediction = predict_dosha(
            st.session_state.uploaded_images,
            st.session_state.user_info["age"], sex_m, model_bundle,
        )

# Dynamically resolve effective markers and apply clinical modulation
if st.session_state.raw_cv_prediction is not None:
    cv_m = st.session_state.get("cv_clinical_markers", {})
    
    # Resolve user overrides vs CV detections
    eff_vein = vein_visibility if vein_visibility != "Auto (from images)" else cv_m.get("vein_visibility", "Low")
    eff_mole = mole_visibility if mole_visibility != "Auto (from images)" else cv_m.get("mole", "No")
    eff_skin = skin_type if skin_type != "Auto (from images)" else cv_m.get("skin_texture", "Smooth")
    eff_sc = skin_color_in if skin_color_in != "Auto (from images)" else cv_m.get("skin_color", "Medium")
    eff_pc = palm_color_in if palm_color_in != "Auto (from images)" else cv_m.get("palm_color", "PinkWhite")
    eff_hair = hair_thickness_in if hair_thickness_in != "Auto (from images)" else cv_m.get("hair_thickness", "Thin")
    eff_hair_c = cv_m.get("hair_color", "Brown")

    is_doctor_mode = any([
        vein_visibility != "Auto (from images)",
        mole_visibility != "Auto (from images)",
        skin_type != "Auto (from images)",
        skin_color_in != "Auto (from images)",
        palm_color_in != "Auto (from images)",
        hair_thickness_in != "Auto (from images)"
    ])

    from unified_cv import apply_clinical_modulation
    st.session_state.prediction = apply_clinical_modulation(
        st.session_state.raw_cv_prediction,
        vein=eff_vein,
        mole=eff_mole,
        skin=eff_skin,
        skin_color=eff_sc,
        palm_color=eff_pc,
        hair_thickness=eff_hair,
        hair_color=eff_hair_c
    )

    st.session_state.effective_markers = {
        "Skin Color": (eff_sc, "Practitioner" if skin_color_in != "Auto (from images)" else "Computer Vision"),
        "Skin Texture": (eff_skin, "Practitioner" if skin_type != "Auto (from images)" else "Computer Vision"),
        "Vein Visibility": (eff_vein, "Practitioner" if vein_visibility != "Auto (from images)" else "Computer Vision"),
        "Mole on Skin": (eff_mole, "Practitioner" if mole_visibility != "Auto (from images)" else "Computer Vision"),
        "Palm Color": (eff_pc, "Practitioner" if palm_color_in != "Auto (from images)" else "Computer Vision"),
        "Hair Thickness": (eff_hair, "Practitioner" if hair_thickness_in != "Auto (from images)" else "Computer Vision"),
        "Hair Color": (eff_hair_c, "Computer Vision")
    }

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
            mode_badge = "🩺 Doctor-in-the-Loop Mode" if is_doctor_mode else "⚡ 100% Automated CV Mode"
            st.caption(f"Confidence: {confidence * 100:.1f}% · {model_bundle.get('model_name', 'Unified ML')} ({mode_badge})")
            if is_doctor_mode:
                active_markers = []
                if vein_visibility != "Auto (from images)":
                    active_markers.append(f"Veins: {vein_visibility}")
                if mole_visibility != "Auto (from images)":
                    active_markers.append(f"Mole: {mole_visibility}")
                if skin_type != "Auto (from images)":
                    active_markers.append(f"Skin: {skin_type}")
                st.info(f"**Clinical Markers Applied:** {', '.join(active_markers)}")

    # 7 Physical Clinical Diagnostic Features Card
    if "effective_markers" in st.session_state:
        with st.expander("📋 Verified Clinical Diagnostic Features (7 Markers)", expanded=True):
            em = st.session_state.effective_markers
            f_cols = st.columns(4)
            for idx, (m_name, (m_val, m_src)) in enumerate(em.items()):
                col_target = f_cols[idx % 4]
                with col_target:
                    src_icon = "🩺" if m_src == "Practitioner" else "⚡"
                    st.metric(label=m_name, value=str(m_val), delta=f"{src_icon} {m_src}", delta_color="off")

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
                    confidence=confidence,
                    age=st.session_state.user_info.get("age", 25),
                    top_class=top_class,
                    secondary_class=ranked[1][0] if len(ranked) > 1 else None,
                    effective_markers=st.session_state.get("effective_markers", {}),
                    is_doctor_mode=is_doctor_mode
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
            st.session_state.raw_cv_prediction = None
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
