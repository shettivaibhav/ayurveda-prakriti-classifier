"""
Script to generate the Ayurveda Prakriti Classifier Code File PDF.
Format strictly matches the reference MCA project code file format:
- Title Header with Name, Roll No, Mentor
- File Structure Overview
- Formatted Code for Core ML, CV, and API Modules
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class SimpleNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(SimpleNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(SimpleNumberedCanvas, self).showPage()
        super(SimpleNumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        # We keep it clean without distracting elements, just clean page layout
        pass

def generate_code_pdf(output_pdf="Ayurveda_Prakriti_Classifier_Code_File.pdf"):
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / output_pdf

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=18
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=18,
        textColor=colors.black,
        spaceAfter=6
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.black
    )

    file_heading = ParagraphStyle(
        'FileHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.8,
        textColor=colors.HexColor("#1a1a1a")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("<b>Ayurveda Prakriti Classifier Code File</b>", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Name:</b> Vaibhav Gajanan Shetti", meta_style))
    story.append(Paragraph("<b>Roll No:</b> 17030925048", meta_style))
    story.append(Paragraph("<b>Mentor:</b> Dr. Krantee Jamdaade", meta_style))
    story.append(Spacer(1, 12))

    # File Structure
    story.append(Paragraph("<b>File Structure</b>", section_heading))
    
    file_structure_text = """ayurveda-prakriti-classifier/
├── mini/
│   ├── app.py                         # Streamlit Web App & Live Inference Portal
│   ├── unified_cv.py                 # 70-D Multi-View CV Biomarker & Quality Extractor
│   ├── palm_color.py                 # Anatomical ROI Color Patch Feature Extractor
│   ├── doctor_portal.py              # Doctor Verification & Retraining Backend
│   ├── doctor_ui.py                  # Streamlit Doctor Verification UI
│   ├── supabasehelper.py             # Supabase Cloud REST, Storage & Gemini API
│   ├── model_color.joblib            # Trained ExtraTrees 4-View Color Model
│   └── unified_prakriti_pipeline.joblib # Unified Standardized Scaler + SVM/ET Pipeline
├── model_v2/
│   ├── models.py                     # PyTorch CNN Backbone (MobileNetV2, 4-view fusion)
│   └── avg/
│       ├── train_color_model.py      # ExtraTrees Classifier Training & Validation
│       ├── train_tabular_model.py    # Balanced Random Forest Tabular Model
│       └── exp_tuned.py              # Nested Cross-Validation Model Selection
├── whatsapp_pipeline/
│   ├── pipeline.py                   # End-to-End Conversational Batch Pipeline
│   ├── parser.py                     # Regex Timestamp & Attachment Extractor
│   ├── extractor.py                  # Metadata & Demographics Entity Extractor
│   ├── view_classifier.py            # Multimodal Vision Hand View Classifier
│   ├── predictor.py                  # Multi-Modal Batch Prakriti Predictor API
│   └── exporter.py                   # Master Clinical Excel Generation Engine
├── dataset/
│   ├── doctor_validation_prakriti_master.xlsx # Clinical Ground Truth Dataset
│   └── Wrist_Dataset/                # Standardized Multi-Angle Hand Image Dataset
├── test_unified_system.py            # Comprehensive E2E System Test Suite
└── requirements.txt                  # Python Project Dependencies"""

    story.append(Preformatted(file_structure_text, code_style))
    story.append(Spacer(1, 14))

    # List of important ML, CV, and API files to include
    code_files = [
        ("mini/palm_color.py", "Anatomical ROI Color Patch Feature Extractor (RGB, CIELAB, Chroma, Redness)"),
        ("mini/unified_cv.py", "Unified Computer Vision Feature Extractor (70-D Biomarkers, Laplacian Variance Quality Check)"),
        ("model_v2/avg/train_color_model.py", "Machine Learning Model Training Pipeline (ExtraTrees, Repeated Stratified K-Fold CV)"),
        ("model_v2/avg/train_tabular_model.py", "Ayurvedic Clinical Tabular Classification (Balanced Random Forest on Clinical Diagnostic Signs)"),
        ("mini/doctor_portal.py", "Doctor Verification Engine & Real-Time Retraining Pipeline (Doctor-in-the-Loop)"),
        ("mini/supabasehelper.py", "Cloud Database & Storage API Helper (Supabase Postgres & Google Gemini Vision)"),
        ("whatsapp_pipeline/pipeline.py", "Multi-Channel WhatsApp Agentic Batch Pipeline (End-to-End Processing Orchestration)"),
        ("whatsapp_pipeline/predictor.py", "Multi-Modal Prakriti Predictor & Diagnostic API"),
        ("mini/app.py", "Streamlit Clinical Inference Web Interface & Prediction Controller")
    ]

    for rel_path, desc in code_files:
        full_path = project_root / rel_path
        if not full_path.exists():
            continue
        
        story.append(PageBreak())
        story.append(Paragraph(f"<b>{rel_path}</b>", file_heading))
        story.append(Paragraph(f"<i>// {desc}</i>", ParagraphStyle('Comment', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, textColor=colors.HexColor("#4b5563"), spaceAfter=6)))
        
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            code_text = f.read()

        # Split code into lines and format cleanly
        lines = code_text.splitlines()
        
        # To keep code focused on important ML/CV/API logic, truncate excessively long repetitive HTML/doc blocks if needed
        # but keep all functional logic
        formatted_code = "\n".join(lines)
        story.append(Preformatted(formatted_code, code_style))

    doc.build(story, canvasmaker=SimpleNumberedCanvas)
    print(f"Successfully generated Code File PDF: {pdf_path}")

if __name__ == "__main__":
    generate_code_pdf()
