import streamlit as st
import tempfile
import os
import io
import sys
import contextlib
import pandas as pd
from pathlib import Path

# Insert project root into system path to resolve pipeline modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import whatsapp_pipeline.parser
import whatsapp_pipeline.extractor
import whatsapp_pipeline.view_classifier
import whatsapp_pipeline.predictor
import whatsapp_pipeline.exporter
import whatsapp_pipeline.pipeline
import whatsapp_pipeline.classifier

# Force reload submodules to clear Streamlit memory caching
importlib.reload(whatsapp_pipeline.parser)
importlib.reload(whatsapp_pipeline.extractor)
importlib.reload(whatsapp_pipeline.view_classifier)
importlib.reload(whatsapp_pipeline.predictor)
importlib.reload(whatsapp_pipeline.exporter)
importlib.reload(whatsapp_pipeline.pipeline)
importlib.reload(whatsapp_pipeline.classifier)

from whatsapp_pipeline.pipeline import WhatsAppAIClassifierPipeline
from whatsapp_pipeline.exporter import append_to_source_dataset
from supabasehelper import get_groq_api_key, get_gemini_api_key

def render_whatsapp_batch_page():
    import importlib
    import whatsapp_pipeline.classifier
    import whatsapp_pipeline.view_classifier
    import whatsapp_pipeline.extractor
    import whatsapp_pipeline.pipeline
    importlib.reload(whatsapp_pipeline.classifier)
    importlib.reload(whatsapp_pipeline.view_classifier)
    importlib.reload(whatsapp_pipeline.extractor)
    importlib.reload(whatsapp_pipeline.pipeline)
    from whatsapp_pipeline.pipeline import WhatsAppAIClassifierPipeline

    if "last_imported_subject_ids" not in st.session_state:
        st.session_state.last_imported_subject_ids = []
        
    st.title("📲 WhatsApp Batch Data Import")
    st.markdown(
        "Upload a `.zip` file of your WhatsApp group or chat export containing subject photos "
        "and their demographic details. The pipeline will automatically parse, classify views, "
        "and synchronize them with your local dataset."
    )
    
    # -------------------------
    # Quick Instructions
    # -------------------------
    with st.expander("ℹ️ How to get your WhatsApp export file", expanded=False):
        st.markdown("""
        1. Open the WhatsApp chat or group on your **mobile phone**.
        2. Tap the menu/contact name -> **Export Chat**.
        3. Select **Include Media** / **Attach Media** (critical for images).
        4. Transfer the resulting `.zip` file to your PC.
        """)
        
    # -------------------------
    # UI Controls (Threshold removed so all users are processed and stored)
    # -------------------------
    threshold = 0.0
    existing_groq_key = get_groq_api_key()
    existing_gemini_key = get_gemini_api_key()
    
    uploaded_file = st.file_uploader(
        "Upload WhatsApp Export ZIP",
        type=["zip"],
        help="Upload the .zip file containing the chat txt and wrist images."
    )
    
    # -------------------------
    # Processing Logic
    # -------------------------
    results = []
    if uploaded_file is not None:
        auto_append = st.checkbox(
            "Automatically append all subjects to master dataset & sync to Supabase",
            value=True,
            help="Enabled by default: All parsed subjects will be saved directly into Wrist_Dataset, the master Excel sheet, and Supabase."
        )
        if st.button("Start Batch Import & Classification", type="primary", use_container_width=True):
            # Save uploaded zip file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_zip_path = Path(tmp.name)
                
            # Temporary output path for results Excel
            out_excel = PROJECT_ROOT / "mini" / "whatsapp_batch_results.xlsx"
            
            # Setup API keys in environment
            old_groq_key = os.environ.get("GROQ_API_KEY")
            if existing_groq_key:
                os.environ["GROQ_API_KEY"] = existing_groq_key
            elif "GROQ_API_KEY" not in os.environ:
                os.environ["GROQ_API_KEY"] = ""
                
            old_gemini_key = os.environ.get("GEMINI_API_KEY")
            if existing_gemini_key:
                os.environ["GEMINI_API_KEY"] = existing_gemini_key
            elif "GEMINI_API_KEY" not in os.environ:
                os.environ["GEMINI_API_KEY"] = ""
                
            # Run pipeline capturing prints to show in UI
            log_output = io.StringIO()
            temp_dir = None
            
            prog_bar = st.progress(0, text="Initializing WhatsApp batch pipeline (0%)...")
            status_text = st.empty()

            def handle_progress(pct, msg):
                clamped_pct = min(max(pct, 0), 100)
                prog_bar.progress(clamped_pct / 100.0, text=f"Progress: {clamped_pct}% – {msg}")
                status_text.markdown(f"⏳ **{clamped_pct}%** · {msg}")

            with st.status("Importing WhatsApp Chat Data...", expanded=True) as status_box:
                handle_progress(1, "Extracting and verifying ZIP contents...")
                
                try:
                    import zipfile
                    import shutil
                    
                    # Create temp dir for zip contents
                    temp_dir = Path(tempfile.mkdtemp(prefix="whatsapp_ui_"))
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        
                    # Find the chat text file inside the extracted files
                    txt_files = list(temp_dir.glob("*.txt"))
                    if not txt_files:
                        raise FileNotFoundError("Could not find any text (.txt) file inside the WhatsApp export zip.")
                        
                    chat_path = txt_files[0]
                    media_path = temp_dir
                    
                    # Copy to project root for remote diagnostics
                    shutil.copy2(chat_path, PROJECT_ROOT / "scratch_uploaded_chat.txt")
                    
                    # Redirect stdout to capture logs in real-time
                    with contextlib.redirect_stdout(log_output):
                        pipeline = WhatsAppAIClassifierPipeline(confidence_threshold=threshold)
                        results = pipeline.process_chat_export(
                            chat_txt_path=chat_path,
                            media_dir_path=media_path,
                            output_excel_path=out_excel,
                            progress_callback=handle_progress
                        )
                        
                        # Sync with main training dataset only if user opted in
                        if auto_append:
                            handle_progress(98, "Appending verified subjects to master Excel & syncing to Supabase...")
                            appended_ids = append_to_source_dataset(
                                results,
                                xlsx_path=PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx",
                                wrist_dataset_dir=PROJECT_ROOT / "dataset" / "Wrist_Dataset"
                            )
                            st.session_state.last_imported_subject_ids = appended_ids
                        else:
                            print("\n[Safe Mode] Master dataset append skipped. Results saved to batch Excel only.")
                        
                    handle_progress(100, "Completed batch import successfully!")
                    status_box.update(label="🎉 Batch Import Complete (100%)!", state="complete", expanded=False)
                except Exception as e:
                    status_box.update(label="❌ Import Failed", state="error", expanded=True)
                    st.error(f"Failed to process chat: {e}")
                    
                # Restore API keys
                if old_groq_key:
                    os.environ["GROQ_API_KEY"] = old_groq_key
                else:
                    os.environ.pop("GROQ_API_KEY", None)
                    
                if old_gemini_key:
                    os.environ["GEMINI_API_KEY"] = old_gemini_key
                else:
                    os.environ.pop("GEMINI_API_KEY", None)
                    
                # Clean up temporary folders & zip
                try:
                    if temp_dir and temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    if tmp_zip_path.exists():
                        os.remove(tmp_zip_path)
                except:
                    pass
            
            # Show logs in expander
            with st.expander("📄 View Import Logs", expanded=False):
                st.code(log_output.getvalue())
                
            # Display Results Summary
            if results:
                st.success("🎉 Batch import successfully completed!")
                
                total = len(results)
                appended_count = len(st.session_state.get("last_imported_subject_ids", [])) if auto_append else total
                
                # Metric display
                m1, m2 = st.columns(2)
                m1.metric("Total Subjects in Archive", total)
                m2.metric("Saved to Dataset & Cloud", appended_count if auto_append else "Pending Commit")
                
                # Flatten for preview table
                preview_data = []
                for idx, r in enumerate(results):
                    k_p = r.get("kapha_prob", 0.0) * 100
                    p_p = r.get("pitta_prob", 0.0) * 100
                    v_p = r.get("vata_prob", 0.0) * 100
                    preview_data.append({
                        "No": idx + 1,
                        "Name": r.get("name", "Unknown"),
                        "Age": r.get("age", "Unknown"),
                        "Gender": r.get("gender", "Unknown"),
                        "Skin Color": r.get("skin_color", "Medium"),
                        "Skin Texture": r.get("skin_texture") or r.get("skin_type", "Smooth"),
                        "Palm Color": r.get("palm_color", "PinkWhite"),
                        "Hair": f"{r.get('hair_thickness', 'Thin')} ({r.get('hair_color', 'Brown')})",
                        "Veins": r.get("vein_visibility", "Low"),
                        "Mole": r.get("mole", "No"),
                        "Predicted Prakriti": r.get("predicted_dosha", "Unknown"),
                        "Confidence": f"{r.get('confidence', 0)*100:.1f}%",
                        "Dosha Split (K/P/V)": f"K-{k_p:.0f}%, P-{p_p:.0f}%, V-{v_p:.0f}%",
                        "Review Required": "⚠️ Yes" if r.get("needs_review", False) else "✅ No"
                    })
                df_preview = pd.DataFrame(preview_data)
                
                st.subheader("Imported Data Preview")
                st.dataframe(df_preview, use_container_width=True, hide_index=True)
                
                # CSV download button
                df_preview_csv = df_preview.to_csv(index=False)
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.download_button(
                        label="📥 Download Import CSV",
                        data=df_preview_csv,
                        file_name="whatsapp_import_metadata.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with c_col2:
                    if not auto_append and total > 0:
                        if st.button(f"📥 Commit & Append All {total} Subjects to Dataset", type="primary", use_container_width=True):
                            with st.spinner("Appending all subjects to palm_dataset_26 June 26.xlsx and Supabase..."):
                                appended_ids = append_to_source_dataset(
                                    results,
                                    xlsx_path=PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx",
                                    wrist_dataset_dir=PROJECT_ROOT / "dataset" / "Wrist_Dataset"
                                )
                                st.session_state.last_imported_subject_ids = appended_ids
                                st.success(f"✅ Successfully appended all {len(appended_ids)} subjects to master training dataset and Supabase!")
                
                # Clean up temporary batch Excel if needed
                try:
                    if out_excel.exists():
                        os.remove(out_excel)
                except Exception:
                    pass
