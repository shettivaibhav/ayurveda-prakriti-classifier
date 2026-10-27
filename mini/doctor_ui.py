"""
Doctor Clinical Validation & Retraining Portal (Streamlit UI)
Provides:
  1. Doctor PIN Authentication.
  2. Visual Review Queue with 4 hand photos side-by-side.
  3. Pre-populated user-given traits (vein, mole, skin texture) + CV estimates.
  4. Single-click 3-way synchronization (Excel, Local Folders, Supabase Cloud).
  5. One-click instant model retraining with live metrics.
"""

import streamlit as st
from PIL import Image
from doctor_portal import get_queue_summary, get_subject_data, save_doctor_verification, retrain_model_pipeline

DOSHA_OPTIONS = [
    "K-P", "P-K", "K-V", "V-K", "P-V", "V-P", "K", "P", "V"
]

DOSHA_EXPLANATIONS = {
    "K-P": "🌊🔥 Kapha dominant, Pitta sub-dominant",
    "P-K": "🔥🌊 Pitta dominant, Kapha sub-dominant (distinct from K-P)",
    "K-V": "🌊🌬️ Kapha dominant, Vata sub-dominant",
    "V-K": "🌬️🌊 Vata dominant, Kapha sub-dominant (distinct from K-V)",
    "P-V": "🔥🌬️ Pitta dominant, Vata sub-dominant",
    "V-P": "🌬️🔥 Vata dominant, Pitta sub-dominant (distinct from P-V)",
    "K": "🌊 Pure Kapha dominant",
    "P": "🔥 Pure Pitta dominant",
    "V": "🌬️ Pure Vata dominant"
}


def render_doctor_portal_page():
    st.markdown("## 🩺 **Doctor Clinical Validation & Training Portal**")
    st.caption("Review participant photos, verify all 7 clinical traits + ground-truth dosha, and retrain the AI model with 1-click.")

    # 1. Doctor PIN Authentication
    if "doctor_authenticated" not in st.session_state:
        st.session_state.doctor_authenticated = False

    if not st.session_state.doctor_authenticated:
        with st.container(border=True):
            st.markdown("### 🔐 Doctor Sign-In")
            pin = st.text_input("Enter Doctor PIN / Access Code:", type="password", placeholder="e.g. 1234")
            if st.button("Unlock Doctor Portal", type="primary"):
                if pin in ["1234", "doctor", "ayurveda", ""]:
                    st.session_state.doctor_authenticated = True
                    st.success("Doctor authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Invalid PIN. (Default: 1234)")
        return

    # Top Navigation / Logout
    cols_top = st.columns([4, 1])
    with cols_top[1]:
        if st.button("🔒 Sign Out"):
            st.session_state.doctor_authenticated = False
            st.rerun()

    # 2. Status & Metric Badges
    pending, verified, duplicates = get_queue_summary()

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Verified Benchmark", f"{len(verified)}", delta="Ground Truth")
    with stat_col2:
        st.metric("Pending Doctor Review", f"{len(pending)}", delta=f"{len(pending)} remaining", delta_color="inverse")
    with stat_col3:
        st.metric("Duplicates Excluded", f"{len(duplicates)}", delta="Flagged")
    with stat_col4:
        st.metric("Active AI Model", "Unified RBF-SVM", delta="Balanced")

    st.markdown("---")

    # Tabs: [1] Review & Verify Queue, [2] One-Click Retraining
    tab_review, tab_retrain = st.tabs(["📋 Review & Verify Queue", "🚀 Retrain AI Model"])

    # ==========================================
    # TAB 1: REVIEW & VERIFY QUEUE
    # ==========================================
    with tab_review:
        queue_mode = st.radio("Select Queue:", ["Pending Review (To-Do)", "Already Verified (Inspect/Edit)"], horizontal=True)
        active_list = pending if queue_mode == "Pending Review (To-Do)" else verified

        if not active_list:
            st.success("🎉 All records in this queue have been verified!")
            return

        # Subject Selector Dropdown
        options = [f"{item['id']} — {item['name'] or 'Unknown'} ({item.get('status', '')})" for item in active_list]
        selected_idx = st.selectbox("Select Participant:", range(len(options)), format_func=lambda i: options[i])
        selected_item = active_list[selected_idx]
        selected_id = selected_item["id"]

        # Load Subject Details
        subj = get_subject_data(selected_id)
        defaults = subj["defaults"]

        # Main Layout: Left 4 Photos, Right Clinical Overwrite Form
        col_photos, col_form = st.columns([1.1, 1.0])

        with col_photos:
            st.markdown(f"#### 🖐️ Hand Photographs ({selected_id})")
            st.caption(f"**Name:** {subj['name']} | **Age:** {subj['age']} | **Sex:** {subj['sex']}")

            imgs = subj["images"]
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                st.caption("1. Palm Open Front (Palm Color)")
                if imgs.get("palm_open_front"):
                    st.image(imgs["palm_open_front"], use_container_width=True)
                else:
                    st.warning("Not available")

            with r1_c2:
                st.caption("2. Palm Open Back (Skin & Veins)")
                if imgs.get("palm_open_back"):
                    st.image(imgs["palm_open_back"], use_container_width=True)
                else:
                    st.warning("Not available")

            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                st.caption("3. Palm Closed Front (Fist)")
                if imgs.get("palm_closed_front"):
                    st.image(imgs["palm_closed_front"], use_container_width=True)
                else:
                    st.warning("Not available")

            with r2_c2:
                st.caption("4. Palm Closed Back (Outer Fist)")
                if imgs.get("palm_closed_back"):
                    st.image(imgs["palm_closed_back"], use_container_width=True)
                else:
                    st.warning("Not available")

        with col_form:
            st.markdown("#### 🩺 Doctor Clinical Overwrite & Verification")

            # Reference AI suggestion
            st.info(f"🤖 **AI Model Predicted:** `{subj['ml_predicted_dosha']}` ({subj['ml_confidence_str'] or 'Estimated'})")

            # Final Verified Dosha (Ground Truth)
            default_dosha_idx = DOSHA_OPTIONS.index(defaults["dosha"]) if defaults["dosha"] in DOSHA_OPTIONS else 0
            doc_prakriti = st.selectbox(
                "Doctor Verified Prakriti (Ground Truth):",
                options=DOSHA_OPTIONS,
                index=default_dosha_idx,
                help="Order matters: K-P is Kapha-dominant, P-K is Pitta-dominant."
            )
            st.caption(f"⭐ **Clinical Meaning:** {DOSHA_EXPLANATIONS.get(doc_prakriti, '')}")

            st.markdown("##### 🔍 7 Clinical Features (Pre-populated from User + CV):")

            # 1. User-given 3 traits
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                vein_opts = ["Low", "Med", "High"]
                v_idx = vein_opts.index(defaults["vein"]) if defaults["vein"] in vein_opts else 0
                doc_vein = st.selectbox("Vein Visibility:", vein_opts, index=v_idx, help="User-reported value pre-filled")

            with c_f2:
                skin_opts = ["Smooth", "Rough", "Dry", "Oily"]
                st_idx = skin_opts.index(defaults["skin_texture"]) if defaults["skin_texture"] in skin_opts else 0
                doc_skin = st.selectbox("Skin Texture:", skin_opts, index=st_idx, help="User-reported value pre-filled")

            with c_f3:
                mole_opts = ["No", "Yes"]
                m_idx = mole_opts.index(defaults["mole"]) if defaults["mole"] in mole_opts else 0
                doc_mole = st.selectbox("Mole on Skin:", mole_opts, index=m_idx, help="User-reported value pre-filled")

            # 2. Remaining 4 traits (pre-filled from CV / doctor default)
            c_f4, c_f5 = st.columns(2)
            with c_f4:
                sc_opts = ["Fair", "Medium", "Dark"]
                sc_idx = sc_opts.index(defaults["skin_color"]) if defaults["skin_color"] in sc_opts else 1
                doc_sc = st.selectbox("Skin Color:", sc_opts, index=sc_idx)

            with c_f5:
                pc_opts = ["PinkWhite", "PinkRed", "Brown"]
                pc_idx = pc_opts.index(defaults["palm_color"]) if defaults["palm_color"] in pc_opts else 0
                doc_pc = st.selectbox("Palm Color:", pc_opts, index=pc_idx)

            c_f6, c_f7 = st.columns(2)
            with c_f6:
                sh_opts = ["Thin", "Med", "Thick"]
                sh_idx = sh_opts.index(defaults["hair_thickness"]) if defaults["hair_thickness"] in sh_opts else 0
                doc_sh = st.selectbox("Hair Thickness:", sh_opts, index=sh_idx)

            with c_f7:
                shc_opts = ["Black", "Brown"]
                shc_idx = shc_opts.index(defaults["hair_color"]) if defaults["hair_color"] in shc_opts else 1
                doc_shc = st.selectbox("Hair Color:", shc_opts, index=shc_idx)

            doc_notes = st.text_input("Doctor Clinical Notes:", value=defaults["notes"] if defaults["notes"] != "Awaiting clinical exam" else "Clinically verified by Doctor")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Confirm & Verify Participant (3-Way Sync)", type="primary", use_container_width=True):
                with st.spinner("Synchronizing to Excel, Local Folders, and Supabase Cloud..."):
                    features_dict = {
                        "vein_visibility": doc_vein,
                        "mole": doc_mole,
                        "skin_texture": doc_skin,
                        "skin_color": doc_sc,
                        "palm_color": doc_pc,
                        "hair_thickness": doc_sh,
                        "hair_color": doc_shc
                    }
                    save_doctor_verification(selected_id, doc_prakriti, features_dict, doc_notes)
                    st.balloons()
                    st.success(f"✅ Successfully verified **{selected_id}** as **{doc_prakriti}**! Synced everywhere.")
                    st.rerun()

    # ==========================================
    # TAB 2: ONE-CLICK MODEL RETRAINING
    # ==========================================
    with tab_retrain:
        st.markdown("### 🚀 One-Click Model Retraining on Ground Truth")
        st.markdown(
            "Retrain the entire machine learning pipeline on **all doctor-verified participants**.\n"
            "This extracts 70-D features from all verified records, fits a class-weighted RBF-SVM, evaluates 5-fold CV, "
            "and immediately replaces the live model in memory."
        )

        st.info(f"📊 **Verified Clinical Training Cohort Available:** `{len(verified)} subjects` (100% verified ground truth)")

        if st.button("⚡ Start AI Model Retraining", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.markdown(f"**Status:** {msg}")

            with st.spinner("Retraining model in progress..."):
                res = retrain_model_pipeline(progress_callback=update_progress)

            if res.get("success"):
                st.balloons()
                st.success(f"✅ Model successfully retrained on **{res['n_samples']} verified participants**!")
                
                # Clear model cache so Streamlit loads the new weights immediately
                st.cache_resource.clear()

                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("5-Fold Balanced Accuracy", f"{res['balanced_accuracy']*100:.1f}%", delta="Honest CV")
                with m_col2:
                    st.metric("5-Fold Raw Accuracy", f"{res['accuracy']*100:.1f}%")
                with m_col3:
                    st.metric("Macro F1-Score", f"{res['macro_f1']*100:.1f}%")

                st.json(res["class_distribution"])
            else:
                st.error(f"Retraining failed: {res.get('error')}")
