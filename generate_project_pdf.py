import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count
    along with running header and footer.
    """
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # Color definitions
        primary_color = colors.HexColor("#1b4332") # Deep Ayurvedic Forest Green
        gray_text = colors.HexColor("#6c757d")
        line_color = colors.HexColor("#d8f3dc")

        # Top Header (Only on page 2 and later)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(gray_text)
            self.drawString(45, 755, "Ayurvedic Prakriti Classifier — Technical & API Overview")
            self.setStrokeColor(line_color)
            self.setLineWidth(0.75)
            self.line(45, 748, 567, 748)

        # Bottom Footer (On all pages)
        self.setFont("Helvetica", 8.5)
        self.setFillColor(gray_text)
        self.drawString(45, 30, "Confidential & Proprietary — Ayurvedic Prakriti Classification System")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 30, page_str)
        
        self.setStrokeColor(line_color)
        self.setLineWidth(0.75)
        self.line(45, 42, 567, 42)
        
        self.restoreState()

def build_pdf(filename="Ayurvedic_Prakriti_Classifier_Comprehensive_Overview.pdf"):
    pdf_path = Path(filename).resolve()
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Color Palette
    PRIMARY = colors.HexColor("#1b4332")     # Deep Forest Green
    SECONDARY = colors.HexColor("#2d6a4f")   # Medium Forest Green
    ACCENT = colors.HexColor("#d4a373")      # Warm Muted Gold
    TEXT_DARK = colors.HexColor("#212529")   # Charcoal
    BG_LIGHT = colors.HexColor("#f8f9fa")    # Off-white / Soft Grey
    CALLOUT_BG = colors.HexColor("#f4f9f4")  # Soft Green Tint
    BORDER_COLOR = colors.HexColor("#b7e4c7")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_LEFT,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        textColor=SECONDARY,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.0,
        leading=13.0,
        alignment=TA_LEFT,
        textColor=TEXT_DARK,
        leftIndent=12,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=TA_LEFT,
        textColor=TEXT_DARK
    )

    code_cell_style = ParagraphStyle(
        'CodeCell',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.0,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=PRIMARY
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.0,
        leading=13.0,
        alignment=TA_LEFT,
        textColor=PRIMARY
    )

    story = []

    # Title & Header Banner
    story.append(Paragraph("AYURVEDIC PRAKRITI CLASSIFIER", title_style))
    story.append(Paragraph("Non-Invasive Body Constitution Assessment via Computer Vision, ML & Agentic AI Pipelines", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=12))

    # 1. Project Title & Overview
    story.append(Paragraph("1. Project Title & Overview", h1_style))
    p_overview = (
        "<b>Project Title:</b> Automated Ayurvedic Prakriti Classifier & Agentic WhatsApp AI Pipeline.<br/>"
        "<b>Overview:</b> In Ayurveda, <i>Prakriti</i> represents an individual's unique biological constitution, categorized "
        "into three primary doshas: <b>Kapha (K)</b>, <b>Pitta (P)</b>, and <b>Vata (V)</b>. Traditionally, identifying a person's "
        "dominant dosha requires lengthy clinical questionnaires and subjective physical observation by experienced practitioners. "
        "This project introduces an end-to-end automated, non-invasive system that classifies a person's dominant dosha using "
        "standardized hand photographs combined with basic demographic parameters (age and sex). The solution includes both an interactive "
        "web-based analytics dashboard (built with Streamlit) and an automated agentic AI pipeline for processing conversational "
        "WhatsApp chat exports."
    )
    story.append(Paragraph(p_overview, body_style))

    # Callout Box for Core Objective
    callout_data = [[
        Paragraph("<b>Core Objective:</b> Democratize and automate traditional Ayurvedic physical assessment through objective computer vision feature extraction (skin & palm color patch analytics) and machine learning, reducing reliance on manual inspection while enabling instant digital record-keeping.", callout_style)
    ]]
    callout_table = Table(callout_data, colWidths=[522])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CALLOUT_BG),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # 2. How the System Works
    story.append(Paragraph("2. How the System Works", h1_style))
    p_works = (
        "The system follows a modular, 5-stage processing workflow spanning user interaction, automated data parsing, "
        "physiological feature extraction, machine learning classification, and cloud database synchronization:"
    )
    story.append(Paragraph(p_works, body_style))

    workflow_steps = [
        "<b>Stage 1: Multi-Modal Data Ingestion:</b> Users upload four standardized hand photographs (wrist/hand front open, back open, front close, back close) along with age and sex via the Streamlit Web Application OR export a WhatsApp chat log containing patient messages and media attachments.",
        "<b>Stage 2: Agentic Chat & Image Parsing (WhatsApp Pipeline):</b> When processing WhatsApp chat exports, an agentic LLM parser extracts structured metadata (subject IDs, age, sex) and uses multimodal AI vision to auto-classify image views (e.g. distinguishing palm-front from back-of-hand).",
        "<b>Stage 3: Automated Physiological Feature Extraction:</b> The system crops standardized central patches from the hand images: skin color from the back-of-hand view and palm color from the front view. It calculates color features across RGB and CIELAB color spaces, Euclidean Chroma (<i>C*</i>), and Redness Index (<i>a*/b*</i>).",
        "<b>Stage 4: Machine Learning Inference:</b> Extracted color vectors are combined with demographic features (age and sex) and passed into a hyperparameter-tuned <b>ExtraTrees (Extremely Randomized Trees) Classifier</b>. The model outputs probability scores for Kapha, Pitta, and Vata doshas.",
        "<b>Stage 5: Synchronization & Visualization:</b> Diagnostic predictions and color metrics are saved to a central cloud database (Supabase), visualized dynamically on the web dashboard (probability gauges/bar charts), and exported as structured Excel dataset reports."
    ]
    for step in workflow_steps:
        story.append(Paragraph(f"• &nbsp; {step}", bullet_style))
    story.append(Spacer(1, 8))

    # 3. Input / Data Collection
    story.append(Paragraph("3. Input / Data Collection", h1_style))
    p_input = (
        "The system operates strictly on minimal, easily accessible inputs to ensure seamless non-invasive deployment:"
    )
    story.append(Paragraph(p_input, body_style))

    input_table_data = [
        [Paragraph("Input Type", table_header_style), Paragraph("Specification / Description", table_header_style), Paragraph("Derived Features / Purpose", table_header_style)],
        [Paragraph("Hand Photographs (4 Views)", code_cell_style), Paragraph("• Front Open (Palm View)<br/>• Back Open (Dorsal View)<br/>• Front Close / Back Close", table_cell_style), Paragraph("Palm pigmentation, skin undertones, CIELAB L*a*b*, Chroma & Redness metrics.", table_cell_style)],
        [Paragraph("Demographic Data", code_cell_style), Paragraph("• Age (Numerical integer)<br/>• Sex (Male / Female categorical)", table_cell_style), Paragraph("Constitutional age baseline and physiological sex variance scaling.", table_cell_style)],
        [Paragraph("Conversational Chat Exports", code_cell_style), Paragraph("WhatsApp exported <code>_chat.txt</code> + media folder / <code>.zip</code> archive", table_cell_style), Paragraph("Unstructured patient dialogue parsing, automatic age/sex/image mapping.", table_cell_style)],
    ]
    t_input = Table(input_table_data, colWidths=[130, 192, 200])
    t_input.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_input)
    story.append(Spacer(1, 10))

    # 4. Prakriti Classification & Machine Learning Architecture
    story.append(Paragraph("4. Prakriti Classification & Machine Learning", h1_style))
    p_prakriti = (
        "In Ayurvedic theory, body constitution is categorized into three primary doshas: <b>Vata</b> (governing movement/dryness), "
        "<b>Pitta</b> (governing metabolism/heat/redness), and <b>Kapha</b> (governing structure/moisture/pallor). "
        "Our empirical research evaluated multiple machine learning paradigms across 92 clinically labeled human subjects:"
    )
    story.append(Paragraph(p_prakriti, body_style))

    story.append(Paragraph("Empirical Model Evaluation & Baseline Comparison:", h2_style))
    ml_points = [
        "<b>Deployed Colour Model (ExtraTrees):</b> Trained on color-patch features (skin + palm RGB/CIELAB/Chroma/Redness) combined with age and sex. Achieves <b>~64% raw accuracy</b> and <b>~53% balanced accuracy</b> (permutation test <i>p ≈ 0.005</i>, statistically significant improvement over the ~64% majority-class Kapha baseline).",
        "<b>Raw-Image Deep Learning (CNN / ViT):</b> Evaluated MobileNetV2, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, and DINOv2-ViT-S backbones. Raw image embeddings performed near chance for this task because uncalibrated phone photos introduce lighting variance that distorts raw pixel representations.",
        "<b>Subject-Grouped Cross-Validation:</b> All evaluations strictly enforce subject-level cross-validation (all 4 views of a subject remain in the same validation fold) to prevent data leakage and ensure realistic real-world performance.",
        "<b>3-Class Dominant vs. 7-Class Dual Prakriti:</b> 3-class dominant dosha (Vata / Pitta / Kapha) provides a reliable clinical signal (~74% upper bound with expert manual features), whereas 7-class dual Prakriti faces a low statistical ceiling (~44%)."
    ]
    for pt in ml_points:
        story.append(Paragraph(f"• &nbsp; {pt}", bullet_style))
    story.append(Spacer(1, 10))

    # Page Break for clean multi-page transition
    story.append(PageBreak())

    # 5. Technology Used & API Architecture (In Simple Words)
    story.append(Paragraph("5. Technology Used & API Architecture", h1_style))
    p_tech = (
        "The project combines modern Python frameworks, computer vision algorithms, machine learning models, cloud databases, "
        "and generative AI APIs. Below is a simple, comprehensive breakdown of every API used, how it works, its role, and how it can be extended:"
    )
    story.append(Paragraph(p_tech, body_style))

    api_table_data = [
        [Paragraph("API / Technology", table_header_style), Paragraph("How It Works & Purpose in Project", table_header_style), Paragraph("Extensibility & Future Potential", table_header_style)],
        
        [Paragraph("<b>Google Gemini API</b><br/><code>google-genai</code>", code_cell_style), 
         Paragraph("<b>Multimodal AI Vision & LLM:</b> Reads raw WhatsApp text messages to extract patient age, sex, and metadata. Uses AI vision to automatically recognize hand image angles (front vs. back views) from unorganized photos.", table_cell_style), 
         Paragraph("Can be extended to automatically respond to patients in WhatsApp, parse medical history, provide Ayurvedic diet tips, or support multi-language dialogue (Hindi, Sanskrit).", table_cell_style)],
        
        [Paragraph("<b>Supabase API</b><br/><code>supabase-py</code>", code_cell_style), 
         Paragraph("<b>Cloud Backend & Relational Database:</b> Connects the web app to a secure PostgreSQL database in the cloud. Automatically stores subject demographics, color metrics, predicted doshas, and timestamps.", table_cell_style), 
         Paragraph("Can be extended with Supabase Storage for secure cloud image hosting, user authentication (login/signup), and real-time WebSockets for instant clinical notifications.", table_cell_style)],
        
        [Paragraph("<b>Streamlit Framework</b><br/><code>streamlit</code>", code_cell_style), 
         Paragraph("<b>Web Interface & User Dashboard:</b> Provides a clean, interactive graphical user interface. Clinicians and users upload hand photos, view real-time dosha percentage gauges, and inspect color analysis.", table_cell_style), 
         Paragraph("Can be extended with custom React components, full REST APIs (via FastAPI integration), downloadable clinical PDF summaries, and practitioner admin portals.", table_cell_style)],

        [Paragraph("<b>Scikit-Learn API</b><br/><code>sklearn</code> / <code>joblib</code>", code_cell_style), 
         Paragraph("<b>Machine Learning Engine:</b> Runs the trained <b>ExtraTreesClassifier</b> model (<code>model_color.joblib</code>). Handles feature scaling, cross-validation, and instant dosha probability calculation.", table_cell_style), 
         Paragraph("Allows effortless swapping or ensembling of ML models (e.g. XGBoost, LightGBM, Neural Networks) as larger clinical datasets are gathered.", table_cell_style)],

        [Paragraph("<b>OpenCV & PIL APIs</b><br/><code>cv2</code> / <code>Pillow</code>", code_cell_style), 
         Paragraph("<b>Computer Vision & Color Analytics:</b> Crops central skin and palm patches from images. Converts RGB colors to CIELAB space and calculates Chroma (<i>sqrt(a*² + b*²)</i>) and Redness Index (<i>a*/b*</i>).", table_cell_style), 
         Paragraph("Can integrate automated hand-landmark detection (MediaPipe Hands) for zero-click palm segmentation and automated color-calibration card correction.", table_cell_style)],

        [Paragraph("<b>Pandas & OpenPyXL</b><br/><code>pandas</code> / <code>openpyxl</code>", code_cell_style), 
         Paragraph("<b>Data Export & Sync API:</b> Structures parsed WhatsApp data and ML outputs into standardized Excel files (<code>whatsapp_prakriti_results.xlsx</code>) and appends to clinical master datasets.", table_cell_style), 
         Paragraph("Can export directly to EHR systems (FHIR format), CSV data pipelines, or automated statistical reporting dashboards.", table_cell_style)],

        [Paragraph("<b>PyTorch & Timm APIs</b><br/><code>torch</code> / <code>timm</code>", code_cell_style), 
         Paragraph("<b>Deep Learning Research Engine:</b> Used in background research scripts (<code>model_v2</code>) to train MobileNetV2, ResNet, and ViT models for 4-view multi-angle image fusion.", table_cell_style), 
         Paragraph("Can be leveraged to fine-tune custom vision-language foundation models when large multi-center clinical image datasets become available.", table_cell_style)],
    ]

    t_api = Table(api_table_data, colWidths=[115, 207, 200])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_api)
    story.append(Spacer(1, 10))

    # 6. System Output
    story.append(Paragraph("6. System Output", h1_style))
    p_output = (
        "The system generates structured, multi-channel outputs for both clinicians and end-users:"
    )
    story.append(Paragraph(p_output, body_style))

    outputs_list = [
        "<b>Dominant Dosha Classification:</b> Clear prediction of the primary constitution (<b>Kapha</b>, <b>Pitta</b>, or <b>Vata</b>).",
        "<b>Dosha Probability Distribution:</b> Percentage breakdown for each dosha (e.g. 55% Kapha, 30% Pitta, 15% Vata) displayed via interactive visual progress bars.",
        "<b>Extracted Color Metrics:</b> Numerical breakdown of skin and palm color spaces (RGB, CIELAB L*a*b*, Chroma, and Redness Index).",
        "<b>Cloud Database Records:</b> Instant insertion into Supabase PostgreSQL tables storing subject ID, age, sex, color parameters, predicted dosha, and timestamp.",
        "<b>Structured Excel Data Reports:</b> Automatically generated <code>whatsapp_prakriti_results.xlsx</code> summarizing batch predictions and patient logs for research and clinical audits."
    ]
    for out in outputs_list:
        story.append(Paragraph(f"• &nbsp; {out}", bullet_style))
    story.append(Spacer(1, 10))

    # 7. Key Features
    story.append(Paragraph("7. Key Features", h1_style))
    features_list = [
        "<b>Fully Automated Image & Demographic Pipeline:</b> Requires only 4 hand photos + age & sex; eliminates tedious clinical questionnaires.",
        "<b>Dual Interface Accessibility:</b> Accessible via an intuitive Streamlit Web Dashboard or a background WhatsApp Conversational Pipeline.",
        "<b>Agentic AI Chat Parsing & Vision Classification:</b> Powered by Gemini API to parse unstructured WhatsApp chats and automatically categorize multi-angle hand photos.",
        "<b>Objective Color-Patch Features:</b> Uses CIELAB, Chroma, and Redness indices extracted from skin and palm areas to minimize subjective human bias.",
        "<b>Cloud Backend & Data Synchronization:</b> Secure real-time cloud persistence powered by Supabase.",
        "<b>Robust Local Fallback Mode:</b> Functions seamlessly even without an active Gemini API key using rule-based token heuristics and chronological ordering.",
        "<b>Research-Grade Validation:</b> Built with strict subject-level cross-validation to guarantee leak-free ML evaluation."
    ]
    for feat in features_list:
        story.append(Paragraph(f"• &nbsp; {feat}", bullet_style))
    story.append(Spacer(1, 10))

    # 8. Future Scope & Roadmap
    story.append(Paragraph("8. Future Scope & Roadmap", h1_style))
    p_future = (
        "The project established a solid proof-of-concept foundation. The following future enhancements will elevate clinical accuracy and scalability:"
    )
    story.append(Paragraph(p_future, body_style))

    future_list = [
        "<b>Standardized Lighting & Color Reference Cards:</b> Incorporating a physical color calibration card (e.g. ColorChecker patch) in photos will normalize ambient lighting, unlocking deeper palm vein and pigmentation signals to boost accuracy from ~64% to >75%.",
        "<b>Automated Hand Landmark Detection (MediaPipe):</b> Replacing static central patch cropping with MediaPipe Hands or YOLOv8-pose to automatically locate palms and dorsal skin regardless of hand position.",
        "<b>Interactive Automated WhatsApp Bot:</b> Deploying a live 24/7 WhatsApp Chatbot (via Meta Graph API / Twilio) that guides patients through step-by-step image capture, collects age/sex interactively, and sends instant Prakriti PDF reports back in chat.",
        "<b>Personalized Ayurvedic Health & Diet Engine:</b> Extending the LLM pipeline to generate customized lifestyle, dietary, and herbal recommendations tailored to the user's predicted dominant dosha.",
        "<b>Mobile App & Clinical EHR Integration:</b> Packaging the solution into React Native or Flutter mobile apps for clinical practitioners with direct export into hospital Electronic Health Record (EHR) systems."
    ]
    for fut in future_list:
        story.append(Paragraph(f"• &nbsp; {fut}", bullet_style))
    story.append(Spacer(1, 14))

    # Sign-off / Metadata footer block
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=5, spaceAfter=8))
    meta_text = "<b>Document Status:</b> Final Project Overview & Technical Specification &nbsp;|&nbsp; <b>Framework:</b> Python, ExtraTrees, Gemini API, Streamlit, Supabase"
    story.append(Paragraph(meta_text, ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor("#6c757d"), alignment=TA_CENTER)))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully created PDF: {pdf_path}")
    return str(pdf_path)

if __name__ == "__main__":
    out1 = build_pdf("Ayurvedic_Prakriti_Classifier_Comprehensive_Overview.pdf")
    # Also update/overwrite Ayurvedic_Prakriti_Classifier_Project_Proposal.pdf so the user's open PDF in VS Code is updated
    out2 = build_pdf("Ayurvedic_Prakriti_Classifier_Project_Proposal.pdf")
