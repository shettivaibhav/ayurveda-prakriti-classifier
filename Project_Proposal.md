# Ayurvedic Prakriti Classification Using Machine Learning

## Introduction:
Ayurveda, the traditional system of medicine, relies on the concept of *Prakriti*—an individual's unique baseline body constitution categorised into three primary doshas: Vata, Pitta, and Kapha. Accurately determining a person's dominant dosha is essential for personalized healthcare, dietary recommendations, and disease prevention. Traditionally, Prakriti assessment requires detailed clinical questionnaires and manual visual inspection by experienced practitioners, which can be time-consuming, subjective, and limited in accessibility for remote or routine health monitoring.

Although modern healthcare applications provide digital diagnostic tools, users often find standalone clinical platforms complex to navigate or inconvenient for frequent routine assessments. To address this challenge, the proposed project aims to develop an automated Ayurvedic Prakriti Classification System integrated with both a web-based analytics dashboard and conversational interfaces such as WhatsApp. The system enables users and practitioners to determine individual dosha constitutions non-invasively through standardized hand photographs and basic demographic data (age and sex).

By leveraging computer vision feature extraction—specifically skin and palm color-patch analytics across color spaces (RGB, CIELAB, chroma, redness indices)—and machine learning classifiers (Extremely Randomized Trees), the system automates physiological feature extraction from non-calibrated image captures. Integrating this predictive model with conversational platforms allows users to submit diagnostic images through everyday messaging applications, making traditional Ayurvedic constitution assessment significantly more accessible, convenient, and efficient.

## Key Features:
• Automated skin and palm color-patch feature extraction (RGB, CIELAB, Chroma, and Redness indices) from multi-view hand photographs.
• Non-invasive prediction of dominant Ayurvedic doshas (Vata, Pitta, Kapha) using tuned machine learning models (ExtraTrees).
• Dual access modes: Interactive web-based analytics dashboard (Streamlit) and automated WhatsApp conversational pipeline.
• Automated image view classification and input validation for standardized multi-angle hand captures.
• Centralized database synchronization (Supabase) for secure patient records and longitudinal history tracking.
• Interactive visualization of dosha probability distributions, visual KPIs, and clinical reports.
• Data export capabilities (Excel, CSV) for research, clinical audits, and statistical analysis.

## Objectives:
The primary objective of the project is to develop a convenient conversational and web interface for Ayurvedic Prakriti classification that reduces dependence on manual clinical inspection for routine body constitution entry. The system aims to simplify record keeping, provide quick access to body constitution diagnostics, and transform collected image and demographic data into meaningful health insights through analytics and visualization.

## Expected Outcome:
The expected outcome is an integrated and scalable system consisting of WhatsApp chatbots, an automated feature extraction engine, a trained ensemble machine learning classifier, and a web-based analytics dashboard. The project will demonstrate how computer vision and conversational interfaces can be integrated with traditional health systems to simplify data collection while providing structured analytics for monitoring and decision-making. The system will be designed as a scalable prototype that can be extended in the future to support additional physiological markers, standardized lighting calibration, and intelligent health recommendations.
