import streamlit as st
import json
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import base64
from io import BytesIO
import uuid
import numpy as np
import cv2
import re
import requests
import os
import textwrap
import tempfile
import threading
import time
import io

# Core ML imports
import torch
import torch.nn as nn
from torchvision import models, transforms

# Audio processing imports with proper error handling
try:
    import sounddevice as sd
    import soundfile as sf
    from pydub import AudioSegment
    import whisper
    WHISPER_AVAILABLE = True
    print("Whisper and audio libraries loaded successfully")
except ImportError as e:
    WHISPER_AVAILABLE = False
    st.warning(f"Whisper or audio libraries not available: {e}")

# PIL for image processing
try:
    from PIL import Image
    import qrcode
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.warning("PIL or qrcode not available")

# Try to import specialized models with fallbacks
try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("transformers not available. Using basic CNN analysis only.")

# Try to import speech recognition with fallback
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

# Translation and TTS imports
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    st.warning("deep_translator not available. Translation features disabled.")

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("gTTS not available. Text-to-speech features disabled.")

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue, red, green
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    PDF_AVAILABLE = True
    print("ReportLab PDF generation available")
except ImportError:
    PDF_AVAILABLE = False
    st.warning("ReportLab not available. PDF export disabled. Install with: pip install reportlab")

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language Configuration
SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "tts_code": "en"},
    "Hindi": {"code": "hi", "tts_code": "hi"},
    "Bengali": {"code": "bn", "tts_code": "bn"},
    "Telugu": {"code": "te", "tts_code": "te"},
    "Marathi": {"code": "mr", "tts_code": "mr"},
    "Tamil": {"code": "ta", "tts_code": "ta"},
    "Gujarati": {"code": "gu", "tts_code": "gu"},
    "Kannada": {"code": "kn", "tts_code": "kn"},
    "Malayalam": {"code": "ml", "tts_code": "ml"},
    "Punjabi": {"code": "pa", "tts_code": "pa"},
    "Odia": {"code": "or", "tts_code": "or"},
    "Urdu": {"code": "ur", "tts_code": "ur"},
    "Assamese": {"code": "as", "tts_code": "as"}
}

# Translation and TTS Manager
class TranslationManager:
    def __init__(self):
        self.available = TRANSLATOR_AVAILABLE
        self.cache = {}
    
    @st.cache_data
    def translate_text(_self, text: str, target_lang_code: str, source_lang_code: str = "auto") -> str:
        """Translate text using Google Translate API"""
        if not _self.available or target_lang_code == "en":
            return text
        
        try:
            cache_key = f"{text}_{source_lang_code}_{target_lang_code}"
            if cache_key in _self.cache:
                return _self.cache[cache_key]
            
            translator = GoogleTranslator(source=source_lang_code, target=target_lang_code)
            translated = translator.translate(text)
            
            _self.cache[cache_key] = translated
            return translated
        except Exception as e:
            st.warning(f"Translation failed: {e}")
            return text
    
    def get_ui_translations(self, selected_lang: str) -> Dict[str, str]:
        """Get UI text translations for selected language"""
        ui_base_texts = {
            "title": "🏥 AI-Powered Health Assistant",
            "subtitle": "Powered by CNN Image Analysis & Whisper Speech Recognition",
            "language_support": "Supporting Multiple Indian Languages",
            "system_status": "System Status",
            "ai_config": "AI Configuration (Optional)",
            "patient_info": "Patient Information",
            "name": "Name",
            "age": "Age", 
            "gender": "Gender",
            "preferred_language": "Preferred Language",
            "medical_history": "Medical History (Optional)",
            "previous_conditions": "Previous Medical Conditions",
            "current_medications": "Current Medications",
            "known_allergies": "Known Allergies", 
            "family_history": "Family Medical History",
            "lifestyle_info": "Lifestyle Information",
            "smoking_status": "Smoking Status",
            "alcohol_consumption": "Alcohol Consumption",
            "exercise_frequency": "Exercise Frequency",
            "stress_level": "Stress Level (1-10)",
            "start_analysis": "🔬 Start CNN-Enhanced Health Analysis",
            "health_consultation": "🔬 CNN-Enhanced Health Consultation",
            "patient_profile": "Patient Profile",
            "ai_analysis_chat": "AI Health Analysis Chat",
            "enhanced_tools": "Enhanced Analysis Tools",
            "voice_input": "🎤 Whisper Voice Input",
            "image_analysis": "📸 CNN Medical Image Analysis",
            "quick_symptoms": "🎯 Quick Symptom Categories",
            "follow_up_questions": "❓ Follow-up Questions",
            "describe_symptoms": "Describe your symptoms in detail...",
            "hello_message": "🤖 Hello! Please describe your symptoms. You can also upload images for CNN analysis or use Whisper voice input.",
            "possible_conditions": "🎯 Possible Conditions",
            "immediate_recommendations": "⚡ Immediate Recommendations", 
            "better_guidance": "❓ To help provide better guidance, could you answer:",
            "important_disclaimer": "⚠️ Important: This analysis incorporates automated CNN image analysis but is for informational purposes only. Always consult a qualified healthcare professional for proper diagnosis and treatment.",
            "diagnosis_english": "Diagnosis (English)",
            "diagnosis_local": "Diagnosis",
            "play_audio": "🔊 Play Audio Diagnosis",
            "translation_error": "Translation service temporarily unavailable",
            # Additional translations for consultation screen
            "male": "Male",
            "female": "Female",
            "other": "Other",
            "prefer_not_to_say": "Prefer not to say",
            "never": "Never",
            "former": "Former",
            "current": "Current",
            "occasional": "Occasional",
            "regular": "Regular",
            "heavy": "Heavy",
            "rarely": "Rarely",
            "weekly": "2-3 times/week",
            "daily": "Daily",
            "respiratory": "Respiratory",
            "heart_chest": "Heart/Chest",
            "digestive": "Digestive",
            "neurological": "Neurological",
            "skin": "Skin",
            "pain": "Pain",
            # PDF Export related translations
            "export_pdf": "📄 Export PDF Summary",
            "generating_pdf": "Generating PDF report...",
            "pdf_generated": "PDF report generated successfully!",
            "pdf_error": "Error generating PDF report"
        }
        
        if selected_lang == "English":
            return ui_base_texts
        
        lang_code = SUPPORTED_LANGUAGES.get(selected_lang, {}).get("code", "en")
        translated_ui = {}
        
        for key, text in ui_base_texts.items():
            translated_ui[key] = self.translate_text(text, lang_code)
        
        return translated_ui

class TextToSpeechManager:
    def __init__(self):
        self.available = TTS_AVAILABLE
        
    def generate_audio(self, text: str, lang_code: str) -> BytesIO:
        """Generate audio from text using gTTS"""
        if not self.available:
            return None
            
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            st.error(f"Text-to-speech generation failed: {e}")
            return None
    
    def play_diagnosis_audio(self, diagnosis_text: str, language: str):
        """Play diagnosis as audio in specified language"""
        if not self.available:
            st.warning("Text-to-speech not available")
            return
        
        lang_info = SUPPORTED_LANGUAGES.get(language, {"tts_code": "en"})
        tts_code = lang_info.get("tts_code", "en")
        
        try:
            audio_buffer = self.generate_audio(diagnosis_text, tts_code)
            if audio_buffer:
                st.audio(audio_buffer, format="audio/mp3")
            else:
                st.error("Could not generate audio")
        except Exception as e:
            st.error(f"Audio playback failed: {e}")

# Enhanced Data Classes
@dataclass
class PatientProfile:
    name: str
    age: int
    gender: str
    language: str
    medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    family_history: List[str]
    lifestyle_factors: Dict[str, str]
    session_id: str

@dataclass
class SymptomAnalysis:
    primary_symptoms: List[str]
    secondary_symptoms: List[str]
    duration: str
    severity: str
    triggers: List[str]
    alleviating_factors: List[str]
    associated_symptoms: List[str]

@dataclass
class DiagnosticResult:
    possible_conditions: List[Dict[str, any]]
    confidence_scores: Dict[str, float]
    recommended_questions: List[str]
    immediate_actions: List[str]
    follow_up_needed: bool
    specialist_referral: Optional[str]

# PDF Generator Class
class PDFReportGenerator:
    def __init__(self, translator: TranslationManager):
        self.available = PDF_AVAILABLE
        self.translator = translator
        
    def generate_health_report(self, patient_profile: PatientProfile, 
                              chat_messages: List, diagnostic_data: Dict,
                              ui_language: str) -> BytesIO:
        """Generate comprehensive health report PDF in both English and patient's language"""
        
        if not self.available:
            raise Exception("PDF generation not available. Install reportlab: pip install reportlab")
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=blue,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=black,
            alignment=TA_LEFT
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.spaceAfter = 6
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph("🏥 AI-Powered Health Assistant - Medical Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
        story.append(Paragraph(f"<b>Session ID:</b> {patient_profile.session_id}", normal_style))
        story.append(Spacer(1, 20))
        
        # Patient Information Section
        story.append(Paragraph("👤 Patient Information", heading_style))
        patient_data = [
            ['Name', patient_profile.name],
            ['Age', str(patient_profile.age)],
            ['Gender', patient_profile.gender],
            ['Preferred Language', patient_profile.language],
            ['UI Language', ui_language]
        ]
        
        if patient_profile.medical_history:
            patient_data.append(['Medical History', ', '.join(patient_profile.medical_history)])
        if patient_profile.medications:
            patient_data.append(['Current Medications', ', '.join(patient_profile.medications)])
        if patient_profile.allergies:
            patient_data.append(['Known Allergies', ', '.join(patient_profile.allergies)])
        if patient_profile.family_history:
            patient_data.append(['Family History', ', '.join(patient_profile.family_history)])
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), '#f0f0f0'),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (0, -1), '#e8e8e8'),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Lifestyle Factors
        if patient_profile.lifestyle_factors:
            story.append(Paragraph("🏃 Lifestyle Information", heading_style))
            lifestyle_data = []
            for key, value in patient_profile.lifestyle_factors.items():
                lifestyle_data.append([key.replace('_', ' ').title(), str(value)])
            
            lifestyle_table = Table(lifestyle_data, colWidths=[2*inch, 4*inch])
            lifestyle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), '#f0f0f0'),
                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            story.append(lifestyle_table)
            story.append(Spacer(1, 20))
        
        # AI Technologies Used
        story.append(Paragraph("🤖 AI Technologies Utilized", heading_style))
        
        # Count technology usage
        cnn_count = sum(1 for msg in chat_messages if 'CNN' in str(msg.content).upper())
        whisper_count = sum(1 for msg in chat_messages if 'WHISPER' in str(msg.content).upper())
        translation_used = patient_profile.language != "English"
        
        tech_data = [
            ['CNN Image Analysis', f"{cnn_count} analyses performed"],
            ['Whisper Speech Recognition', f"{whisper_count} voice inputs processed"],
            ['Translation Service', 'Yes' if translation_used else 'No'],
            ['Text-to-Speech', 'Available' if TTS_AVAILABLE else 'Not used'],
            ['Total AI Interactions', str(len(chat_messages))]
        ]
        
        tech_table = Table(tech_data, colWidths=[3*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), '#e8f4fd'),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 20))
        
        # Extract diagnostic information
        all_conditions = []
        all_recommendations = set()
        
        for msg in chat_messages:
            if hasattr(msg, 'diagnostic_data') and msg.diagnostic_data:
                data = msg.diagnostic_data
                if 'possible_conditions' in data:
                    all_conditions.extend(data['possible_conditions'])
                if 'immediate_actions' in data:
                    all_recommendations.update(data['immediate_actions'])
        
        # Diagnosis Summary (English)
        story.append(Paragraph("🎯 Diagnosis Summary (English)", heading_style))
        
        if all_conditions:
            # Get top conditions by confidence
            condition_scores = {}
            for condition in all_conditions:
                name = condition['name']
                confidence = condition.get('confidence', 0)
                condition_scores[name] = max(condition_scores.get(name, 0), confidence)
            
            sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
            
            conditions_text = "Based on the AI analysis, the following conditions were identified:\n\n"
            for i, (condition_name, confidence) in enumerate(sorted_conditions[:5], 1):
                conditions_text += f"{i}. {condition_name} - {confidence*100:.0f}% likelihood\n"
            
            story.append(Paragraph(conditions_text, normal_style))
        else:
            story.append(Paragraph("No specific conditions identified in this consultation.", normal_style))
        
        story.append(Spacer(1, 15))
        
        # Recommendations (English)
        story.append(Paragraph("💡 Recommendations (English)", heading_style))
        if all_recommendations:
            rec_text = "The following recommendations were provided:\n\n"
            for i, rec in enumerate(list(all_recommendations)[:10], 1):
                rec_text += f"• {rec}\n"
            story.append(Paragraph(rec_text, normal_style))
        else:
            story.append(Paragraph("No specific recommendations were generated.", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Translated sections (if patient language is not English)
        if patient_profile.language != "English" and self.translator.available:
            try:
                lang_code = SUPPORTED_LANGUAGES.get(patient_profile.language, {}).get("code", "hi")
                
                # Translate diagnosis
                story.append(Paragraph(f"🎯 Diagnosis Summary ({patient_profile.language})", heading_style))
                
                if all_conditions:
                    translated_conditions = self.translator.translate_text(conditions_text, lang_code)
                    story.append(Paragraph(translated_conditions, normal_style))
                else:
                    no_conditions_text = "No specific conditions identified in this consultation."
                    translated_no_conditions = self.translator.translate_text(no_conditions_text, lang_code)
                    story.append(Paragraph(translated_no_conditions, normal_style))
                
                story.append(Spacer(1, 15))
                
                # Translate recommendations
                story.append(Paragraph(f"💡 Recommendations ({patient_profile.language})", heading_style))
                if all_recommendations:
                    translated_rec = self.translator.translate_text(rec_text, lang_code)
                    story.append(Paragraph(translated_rec, normal_style))
                else:
                    no_rec_text = "No specific recommendations were generated."
                    translated_no_rec = self.translator.translate_text(no_rec_text, lang_code)
                    story.append(Paragraph(translated_no_rec, normal_style))
                
                story.append(Spacer(1, 20))
                
            except Exception as e:
                story.append(Paragraph(f"Translation error: {str(e)}", normal_style))
                story.append(Spacer(1, 10))
        
        # Consultation Timeline
        story.append(Paragraph("📅 Consultation Timeline", heading_style))
        
        if chat_messages:
            timeline_text = f"Consultation started: {chat_messages[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            timeline_text += f"Total interactions: {len(chat_messages)}\n"
            timeline_text += f"Duration: {(datetime.datetime.now() - chat_messages[0].timestamp).seconds // 60} minutes\n"
            timeline_text += f"User messages: {len([m for m in chat_messages if m.role == 'user'])}\n"
            timeline_text += f"AI responses: {len([m for m in chat_messages if m.role == 'assistant'])}"
            
            story.append(Paragraph(timeline_text, normal_style))
        
        story.append(Spacer(1, 30))
        
        # Medical Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=red,
            alignment=TA_JUSTIFY,
            borderColor=red,
            borderWidth=1,
            borderPadding=10
        )
        
        disclaimer_text = """
        <b>⚠️ MEDICAL DISCLAIMER:</b> This report is generated by an AI-powered health assistant that incorporates 
        CNN image analysis, Whisper speech recognition, and automated translation services. The analysis and 
        recommendations provided are for informational purposes only and should not be considered as professional 
        medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper 
        medical diagnosis and treatment. The AI system cannot replace professional medical judgment and should be 
        used as a supplementary tool only.
        """
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = f"Report generated by AI Health Assistant | Session: {patient_profile.session_id} | {report_date}"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=blue,
            alignment=TA_CENTER
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def create_download_link(self, pdf_buffer: BytesIO, filename: str) -> str:
        """Create download link for PDF"""
        b64 = base64.b64encode(pdf_buffer.read()).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF Report</a>'

# Fixed CNN-based Medical Image Analyzer
class CNNMedicalImageAnalyzer:
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize the CNN analyzer and optional API.
        """
        self.api_key = api_key or "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"
        self.api_url = api_url or "https://api-inference.huggingface.co/models/syaha/skin_cancer_detection_model"
    
        self.cnn_model = None
        self.transform = None
        self.available = False
        
        try:
            self._initialize_models()
            self.available = True
        except Exception as e:
            st.warning(f"CNN models initialization failed: {e}")
    
    def _initialize_models(self):
        """Initialize ResNet50 CNN for general feature extraction."""
        try:
            self.cnn_model = models.resnet50(pretrained=True)
            self.cnn_model.fc = nn.Identity()  # Remove classification head
            self.cnn_model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            st.info("ResNet50 CNN initialized successfully!")
        except Exception as e:
            raise Exception(f"CNN model initialization failed: {e}")
    
    def analyze_medical_image(self, image: Image.Image) -> Dict[str, any]:
        """Analyze an image using CNN and optional skin lesion API."""
        if not self.available:
            return {
                'error': 'CNN model not available',
                'symptom_text': 'Image analysis requires proper model initialization',
                'auto_symptoms': ''
            }
        
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            analysis = {
                'image_quality': self._assess_image_quality(image),
                'cnn_features': self._extract_cnn_features(image),
                'auto_extracted': True
            }
            
            # Use API if configured
            if self.api_key and self.api_url:
                skin_analysis = self._analyze_skin_lesion_api(image)
                if skin_analysis and 'error' not in skin_analysis:
                    analysis['specialized_analysis'] = skin_analysis
                    analysis['symptom_text'] = self._create_symptom_text_from_specialized(skin_analysis)
                else:
                    analysis['symptom_text'] = self._create_symptom_text_from_cnn(analysis['cnn_features'])
            else:
                analysis['symptom_text'] = self._create_symptom_text_from_cnn(analysis['cnn_features'])
            
            analysis['auto_symptoms'] = analysis['symptom_text']
            return analysis
        
        except Exception as e:
            return {
                'error': f"Image analysis failed: {str(e)}",
                'symptom_text': 'Unable to analyze image due to processing error',
                'auto_symptoms': ''
            }
    
    def _assess_image_quality(self, image: Image.Image) -> Dict[str, any]:
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            quality = 'good' if blur_score > 100 and 50 < brightness < 200 else 'fair' if blur_score > 50 else 'poor'
            return {
                'blur_score': f"{blur_score:.2f}",
                'brightness': f"{brightness:.2f}",
                'contrast': f"{contrast:.2f}",
                'assessment': quality
            }
        except Exception as e:
            return {'assessment': 'unknown', 'error': str(e)}
    
    def _extract_cnn_features(self, image: Image.Image) -> Dict[str, any]:
        try:
            img_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.cnn_model(img_tensor).squeeze().numpy()
            
            feature_analysis = self._interpret_cnn_features(features, image)
            
            return {
                'feature_vector': features.tolist()[:20],
                'visual_characteristics': feature_analysis,
                'confidence': 'medium'
            }
        except Exception as e:
            return {'error': f"Feature extraction failed: {e}"}
    
    def _interpret_cnn_features(self, features: np.ndarray, image: Image.Image) -> Dict[str, str]:
        img_array = np.array(image)
        mean_color = np.mean(img_array, axis=(0,1))
        r, g, b = mean_color
        
        characteristics = {}
        if r > g + 20 and r > b + 20:
            characteristics['color'] = 'predominantly reddish coloration observed'
        elif g > r + 10 and g > b + 10:
            characteristics['color'] = 'greenish or yellowish tint detected'
        elif b > r + 10 and b > g + 10:
            characteristics['color'] = 'bluish or purplish discoloration noted'
        else:
            characteristics['color'] = 'normal skin tone range'
        
        texture_variance = np.var(features)
        if texture_variance > 0.5:
            characteristics['texture'] = 'irregular texture or surface variation detected'
        elif texture_variance > 0.2:
            characteristics['texture'] = 'moderate texture variation observed'
        else:
            characteristics['texture'] = 'smooth or uniform texture'
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            characteristics['boundaries'] = 'well-defined borders or lesion boundaries'
        elif edge_density > 0.05:
            characteristics['boundaries'] = 'moderately defined features'
        else:
            characteristics['boundaries'] = 'diffuse or poorly defined boundaries'
        
        return characteristics
    
    def _analyze_skin_lesion_api(self, image: Image.Image) -> Dict[str, any]:
        """Send image to a skin lesion API and return prediction."""
        if not self.api_key or not self.api_url:
            return {'error': 'API key or URL not configured'}
        
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            buffered.seek(0)
            
            files = {"file": ("image.jpg", buffered, "image/jpeg")}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(self.api_url, files=files, headers=headers, timeout=30)
            if response.status_code != 200:
                return {'error': f"API returned status code {response.status_code}: {response.text}"}
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Handle Hugging Face format
                predictions = result
                top_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                return {
                    'primary_prediction': top_prediction.get('label', 'unknown'),
                    'confidence': top_prediction.get('score', 0),
                    'top_predictions': [{'condition': p.get('label'), 'probability': p.get('score')} for p in predictions[:3]]
                }
            
            return result
        except Exception as e:
            return {'error': f"API call failed: {e}"}
    
    def _create_symptom_text_from_specialized(self, skin_analysis: Dict) -> str:
        if 'error' in skin_analysis:
            return "Specialized skin analysis encountered an error"
        
        primary = skin_analysis.get('primary_prediction', 'unknown condition')
        confidence = skin_analysis.get('confidence', 0)
        
        symptom_text = f"API dermatology analysis suggests: {primary} (confidence: {confidence:.2f}). "
        
        top_preds = skin_analysis.get('top_predictions', [])
        if top_preds and len(top_preds) > 1:
            other_possibilities = ", ".join([p['condition'] for p in top_preds[1:]])
            if other_possibilities:
                symptom_text += f"Other possibilities include: {other_possibilities}. "
        
        symptom_text += "This automated analysis should be confirmed by a medical professional."
        return symptom_text
    
    def _create_symptom_text_from_cnn(self, cnn_features: Dict) -> str:
        if 'error' in cnn_features:
            return "CNN image analysis encountered an error"
        
        characteristics = cnn_features.get('visual_characteristics', {})
        symptom_parts = ["Visual analysis from uploaded image:"]
        for desc in characteristics.values():
            symptom_parts.append(desc)
        symptom_parts.append("CNN-based feature analysis completed. Professional medical evaluation recommended.")
        return " ".join(symptom_parts)

# Fixed Whisper Speech Recognizer
class WhisperSpeechRecognizer:
    def __init__(self):
        self.whisper_model = None
        self.available = False
        
        if not WHISPER_AVAILABLE:
            st.warning("Whisper dependencies not available. Please install: pip install openai-whisper sounddevice soundfile pydub")
            return
            
        try:
            # Load Whisper model (base is a good balance of speed/accuracy)
            self.whisper_model = whisper.load_model("base")
            self.available = True
            st.info("Whisper speech recognition model loaded successfully!")
        except Exception as e:
            st.warning(f"Whisper model loading failed: {e}")
            # Fallback to speech_recognition if available
            if SPEECH_AVAILABLE:
                try:
                    self.recognizer = sr.Recognizer()
                    self.microphone = sr.Microphone()
                    self.available = True
                    st.info("Using speech_recognition as fallback")
                except Exception:
                    self.available = False

    def _convert_to_wav(self, audio_bytes: bytes) -> str:
        """Force convert any audio bytes to valid PCM WAV"""
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            # Convert to mono, 16kHz, 16-bit PCM
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio.export(tmp.name, format="wav", codec="pcm_s16le")
            return tmp.name
        except Exception as e:
            raise Exception(f"Audio conversion failed: {e}")
    
    def record_audio(self, duration: int, samplerate: int = 16000) -> str:
        """Record audio and save as valid PCM WAV"""
        try:
            st.info(f"🎤 Recording for {duration} seconds... Speak now!")
            
            recording = sd.rec(
                int(duration * samplerate), 
                samplerate=samplerate, 
                channels=1, 
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, recording, samplerate, subtype="PCM_16")
                return tmp.name
                
        except Exception as e:
            raise Exception(f"Audio recording failed: {e}")
    
    def transcribe_with_whisper(self, duration: int, language: str = None) -> dict:
        """Record and transcribe audio using Whisper"""
        
        if not self.available or not self.whisper_model:
            return {"success": False, "error": "Whisper model not available"}
        
        temp_file = None
        try:
            # Record audio to proper WAV
            temp_file = self.record_audio(duration)
            
            # Transcribe with Whisper
            language_code = self._get_whisper_language_code(language)
            
            st.info("🔄 Transcribing with Whisper...")
            result = self.whisper_model.transcribe(
                temp_file, 
                language=language_code,
                fp16=False
            )
            
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "confidence": "high"
            }
                
        except Exception as e:
            return {"success": False, "error": f"Whisper transcription failed: {e}"}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def transcribe_from_mic_recorder(self, audio_data, language: str = "English") -> dict:
        """Transcribe audio from streamlit-mic-recorder using Whisper"""
        
        if not self.available:
            return {"success": False, "error": "Speech recognition not available"}
        
        temp_file_path = None
        try:
            # Handle mic_recorder audio data
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                audio_bytes = audio_data['bytes']
            elif isinstance(audio_data, (bytes, bytearray)):
                audio_bytes = audio_data
            else:
                return {"success": False, "error": "Invalid audio data format"}
            
            if len(audio_bytes) < 1000:  # Too short
                return {"success": False, "error": "Audio recording too short"}
            
            # Convert to PCM WAV
            temp_file_path = self._convert_to_wav(audio_bytes)
            
            if self.whisper_model:
                # Use Whisper
                language_code = self._get_whisper_language_code(language)
                result = self.whisper_model.transcribe(
                    temp_file_path,
                    language=language_code,
                    fp16=False
                )
                
                text = result["text"].strip()
                detected_language = result.get("language", language)
                
                if not text:
                    return {"success": False, "error": "No speech detected in audio"}
                
            elif SPEECH_AVAILABLE:
                # Fallback to speech_recognition
                with sr.AudioFile(temp_file_path) as source:
                    audio = self.recognizer.record(source)
                
                language_code = self._get_sr_language_code(language)
                text = self.recognizer.recognize_google(audio, language=language_code)
                detected_language = language
                
                if not text:
                    return {"success": False, "error": "No speech detected in audio"}
            else:
                return {"success": False, "error": "No speech recognition backend available"}
            
            return {
                "success": True, 
                "text": text, 
                "language": detected_language
            }
                
        except Exception as e:
            return {"success": False, "error": f"Transcription failed: {e}"}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def _get_whisper_language_code(self, language: str) -> str:
        """Map language names to Whisper language codes"""
        language_mapping = {
            "English": "en",
            "Hindi": "hi",
            "Bengali": "bn", 
            "Telugu": "te",
            "Marathi": "mr",
            "Tamil": "ta",
            "Gujarati": "gu",
            "Urdu": "ur",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa",
            "Odia": "or",
            "Assamese": "as",
            "Spanish": "es",
            "French": "fr",
            "German": "de"
        }
        return language_mapping.get(language, "en")
    
    def _get_sr_language_code(self, language: str) -> str:
        """Map language names to speech_recognition language codes"""
        language_mapping = {
            "English": "en-IN",
            "Hindi": "hi-IN",
            "Bengali": "bn-IN",
            "Telugu": "te-IN",
            "Marathi": "mr-IN",
            "Tamil": "ta-IN",
            "Gujarati": "gu-IN",
            "Urdu": "ur-IN",
            "Kannada": "kn-IN",
            "Malayalam": "ml-IN",
            "Punjabi": "pa-IN",
            "Odia": "or-IN",
            "Assamese": "as-IN"
        }
        return language_mapping.get(language, "en-IN")

# AI-Powered Medical Knowledge System
class AIHealthAnalyzer:
    def __init__(self, openai_key=None, gemini_api_key=None):
        self.openai_client = None
        self.gemini_client = None
        
        # Try to initialize AI clients safely
        try:
            if openai_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
        except Exception as e:
            st.warning("OpenAI client initialization failed")
        
        try:
            if gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai
        except Exception as e:
            st.warning("Gemini client initialization failed")
        
        # Medical knowledge prompts
        self.system_prompts = {
            'symptom_analysis': """
            You are an expert medical AI assistant. Analyze the provided symptoms and patient information.
            If visual analysis from an image is included, integrate it into your diagnosis.
            Always provide:
            1. List of possible conditions (with confidence percentages)
            2. Questions to ask for better diagnosis
            3. Immediate recommendations
            4. Risk assessment (low/medium/high)
            5. When to seek emergency care
            
            Consider: age, gender, medical history, symptom duration, severity, and visual findings from images.
            Always include medical disclaimers about consulting healthcare professionals.
            """,
        }
    
    def analyze_symptoms_with_ai(self, symptoms: str, patient: PatientProfile, 
                                 previous_responses: List[str] = None) -> DiagnosticResult:
        """Use AI to analyze symptoms and generate diagnosis suggestions"""
        
        # Build context for AI
        context = self._build_patient_context(patient, symptoms, previous_responses)
        
        # Try different AI services with fallback
        try:
            if self.openai_client:
                result = self._analyze_with_openai(context)
            elif self.gemini_client:
                result = self._analyze_with_gemini(context)
            else:
                # Use rule-based fallback
                result = self._simulate_ai_response(context)
            
            return self._parse_ai_response(result)
            
        except Exception as e:
            st.warning(f"AI analysis failed, using fallback: {e}")
            return self._fallback_analysis(symptoms, patient)
    
    def _build_patient_context(self, patient: PatientProfile, symptoms: str, 
                               previous_responses: List[str] = None) -> str:
        """Build comprehensive context for AI analysis"""
        
        context = f"""
        PATIENT INFORMATION:
        - Age: {patient.age}
        - Gender: {patient.gender}
        - Language: {patient.language}
        - Medical History: {', '.join(patient.medical_history) if patient.medical_history else 'None reported'}
        - Current Medications: {', '.join(patient.medications) if patient.medications else 'None reported'}
        - Known Allergies: {', '.join(patient.allergies) if patient.allergies else 'None reported'}
        - Family History: {', '.join(patient.family_history) if patient.family_history else 'None reported'}
        
        CURRENT SYMPTOMS (including CNN image analysis):
        {symptoms}
        
        LIFESTYLE FACTORS:
        {json.dumps(patient.lifestyle_factors, indent=2) if patient.lifestyle_factors else 'None provided'}
        """
        
        if previous_responses:
            context += f"\n\nPREVIOUS RESPONSES TO QUESTIONS:\n"
            context += "\n".join(previous_responses)
        
        return context
    
    def _analyze_with_openai(self, context: str) -> str:
        """Analyze using OpenAI GPT"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompts['symptom_analysis']},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    
    def _analyze_with_gemini(self, context: str) -> str:
        """Analyze using Google Gemini"""
        model = self.gemini_client.GenerativeModel('gemini-pro')
        response = model.generate_content(
            self.system_prompts['symptom_analysis'] + "\n\n" + context
        )
        return response.text
    
    def _simulate_ai_response(self, context: str) -> str:
        """Enhanced simulation of AI response with CNN integration"""
        # Extract key information
        age_match = re.search(r'Age: (\d+)', context)
        gender_match = re.search(r'Gender: (\w+)', context)
        symptoms_section = re.search(r'CURRENT SYMPTOMS.*:\n(.*?)(?:\n\n|\Z)', context, re.DOTALL)
        
        age = int(age_match.group(1)) if age_match else 30
        gender = gender_match.group(1) if gender_match else "Unknown"
        symptoms = symptoms_section.group(1).strip() if symptoms_section else "No symptoms provided"
        
        # Enhanced symptom analysis with CNN integration
        symptom_keywords = {
            'cnn_skin': ['CNN', 'dermatology', 'lesion', 'skin', 'visual analysis'],
            'fever': ['fever', 'temperature', 'hot', 'chills'],
            'cough': ['cough', 'coughing', 'throat'],
            'headache': ['headache', 'head pain', 'migraine'],
            'stomach': ['stomach', 'nausea', 'vomit', 'digestive'],
            'chest': ['chest pain', 'heart', 'breathing'],
            'skin': ['rash', 'itch', 'skin', 'spots', 'red', 'swelling', 'bump']
        }
        
        detected_categories = []
        symptoms_lower = symptoms.lower()
        
        for category, keywords in symptom_keywords.items():
            if any(keyword in symptoms_lower for keyword in keywords):
                detected_categories.append(category)
        
        # Generate contextual response with CNN awareness
        response = f"""
        MEDICAL ANALYSIS:
        
        Based on the patient's presentation (Age: {age}, Gender: {gender}), I have analyzed the reported symptoms.
        """
        
        if 'cnn_skin' in detected_categories:
            response += f"""
        
        **CNN IMAGE ANALYSIS DETECTED:** The analysis includes automated visual assessment of skin or tissue.
        
        DETECTED SYMPTOM CATEGORIES: {', '.join(detected_categories)}
        
        POSSIBLE CONDITIONS:
        1. **Dermatological Condition (CNN Detected)** (Confidence: 70%)
           - CNN analysis suggests skin-related findings
           - Visual pattern recognition indicates potential lesion or inflammation
        
        2. **Inflammatory Response** (Confidence: 25%)
           - May be related to immune system activation
           - Consider environmental or contact triggers
        
        3. **Benign Skin Variation** (Confidence: 15%)
           - Could be normal skin variation or minor irritation
           - Monitor for changes over time
        """
        else:
            response += f"""
        
        DETECTED SYMPTOM CATEGORIES: {', '.join(detected_categories) if detected_categories else 'General symptoms'}
        
        POSSIBLE CONDITIONS:
        1. **Viral Upper Respiratory Infection** (Confidence: 60%)
           - Common presentation matching reported symptoms
           - Typically self-limiting with supportive care
        
        2. **Bacterial Infection** (Confidence: 25%)
           - Consider if symptoms persist or worsen
           - May require antibiotic treatment
        
        3. **Allergic Reaction** (Confidence: 15%)
           - Environmental or food-related triggers possible
           - Consider recent exposures or changes
        """
        
        response += """
        RECOMMENDED QUESTIONS:
        1. How long have these symptoms been present?
        2. Have you had any fever? What was the highest temperature?
        3. Are there any triggers you've noticed?
        
        IMMEDIATE RECOMMENDATIONS:
        - Rest and maintain adequate hydration
        - Monitor symptoms and temperature
        - Use appropriate over-the-counter remedies as needed
        - If CNN skin analysis was performed, avoid irritating the affected area
        
        RISK ASSESSMENT: LOW to MEDIUM
        
        WHEN TO SEEK EMERGENCY CARE:
        - High fever (>101.3°F/38.5°C) persisting >3 days
        - Difficulty breathing or chest pain
        - Severe persistent headache
        - Rapid spreading of skin lesions (if applicable)
        
        FOLLOW-UP: Consult healthcare provider if symptoms persist beyond 7-10 days or worsen.
        
        DISCLAIMER: This analysis incorporates automated CNN image recognition but is for informational purposes only. Always consult qualified healthcare professionals for proper diagnosis and treatment.
        """
        
        return response
    
    def _parse_ai_response(self, ai_response: str) -> DiagnosticResult:
        """Parse AI response into structured format"""
        
        # Extract possible conditions
        conditions = []
        condition_pattern = r'(\d+)\.\s*\*\*(.*?)\*\*\s*\(Confidence:\s*(\d+)%\)'
        matches = re.findall(condition_pattern, ai_response)
        
        for match in matches:
            conditions.append({
                'name': match[1].strip(),
                'confidence': float(match[2]) / 100,
                'description': self._extract_condition_description(ai_response, match[1])
            })
        
        # Extract questions
        questions = self._extract_list_items(ai_response, 'RECOMMENDED QUESTIONS:', 'IMMEDIATE')
        
        # Extract immediate actions
        actions = self._extract_list_items(ai_response, 'IMMEDIATE RECOMMENDATIONS:', 'RISK')
        
        # Determine follow-up need
        follow_up_needed = any(phrase in ai_response.lower() for phrase in [
            'consult', 'see doctor', 'medical attention', 'healthcare provider'
        ])
        
        # Extract specialist referral
        specialist_pattern = r'(cardiologist|neurologist|gastroenterologist|dermatologist|pulmonologist|psychiatrist)'
        specialist_match = re.search(specialist_pattern, ai_response.lower())
        specialist = specialist_match.group(1) if specialist_match else None
        
        return DiagnosticResult(
            possible_conditions=conditions,
            confidence_scores={c['name']: c['confidence'] for c in conditions},
            recommended_questions=questions[:5],
            immediate_actions=actions,
            follow_up_needed=follow_up_needed,
            specialist_referral=specialist
        )
    
    def _extract_list_items(self, text: str, start_marker: str, end_marker: str) -> List[str]:
        """Extract list items between markers"""
        try:
            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker, start_idx) if end_marker else len(text)
            
            if start_idx == -1:
                return []
            
            section = text[start_idx:end_idx]
            lines = section.split('\n')
            
            items = []
            for line in lines[1:]:  # Skip the marker line
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                    # Clean the line
                    cleaned = re.sub(r'^[-•\d\.\s]+', '', line).strip()
                    if cleaned:
                        items.append(cleaned)
            
            return items
        except:
            return []
    
    def _extract_condition_description(self, text: str, condition_name: str) -> str:
        """Extract description for a specific condition"""
        try:
            pattern = rf'\*\*{re.escape(condition_name)}\*\*.*?\n(.*?)(?:\n\d+\.|\n\n|$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                description = match.group(1).strip()
                # Clean up bullet points
                description = re.sub(r'^\s*[-•]\s*', '', description, flags=re.MULTILINE)
                # Limit length
                return description[:200] + "..." if len(description) > 200 else description
            return "Description not available"
        except:
            return "Description not available"
    
    def _fallback_analysis(self, symptoms: str, patient: PatientProfile) -> DiagnosticResult:
        """Enhanced fallback analysis with CNN awareness"""
        symptoms_lower = symptoms.lower()
        
        # More comprehensive symptom mapping including CNN findings
        condition_mapping = {
            'cnn|dermatology|skin lesion|visual analysis': {
                'name': 'Dermatological Condition (CNN Detected)',
                'confidence': 0.75,
                'description': 'CNN image analysis suggests skin-related findings requiring evaluation'
            },
            'chest pain|heart|cardiac': {
                'name': 'Cardiac Evaluation Needed',
                'confidence': 0.8,
                'description': 'Chest pain requires immediate cardiac evaluation'
            },
            'fever|temperature|chills': {
                'name': 'Febrile Illness',
                'confidence': 0.7,
                'description': 'Fever suggests infectious or inflammatory process'
            },
            'cough|throat|respiratory': {
                'name': 'Respiratory Infection',
                'confidence': 0.6,
                'description': 'Upper respiratory symptoms suggest viral or bacterial infection'
            },
            'headache|migraine|head pain': {
                'name': 'Headache Disorder',
                'confidence': 0.6,
                'description': 'Head pain may have various underlying causes'
            },
            'stomach|nausea|vomit|digestive': {
                'name': 'Gastrointestinal Issue',
                'confidence': 0.6,
                'description': 'Digestive symptoms require evaluation'
            }
        }
        
        conditions = []
        for pattern, condition_data in condition_mapping.items():
            if re.search(pattern, symptoms_lower):
                conditions.append(condition_data)
        
        # Default condition if none matched
        if not conditions:
            conditions.append({
                'name': 'General Medical Evaluation',
                'confidence': 0.5,
                'description': 'Symptoms require professional medical assessment'
            })
        
        return DiagnosticResult(
            possible_conditions=conditions,
            confidence_scores={c['name']: c['confidence'] for c in conditions},
            recommended_questions=[
                "How long have you experienced these symptoms?",
                "On a scale of 1-10, how severe is your discomfort?",
                "Are you currently taking any medications?",
            ],
            immediate_actions=[
                "Monitor symptoms closely",
                "Stay hydrated and rest",
                "Seek medical attention if symptoms worsen",
                "If CNN analysis detected skin changes, avoid further irritation",
            ],
            follow_up_needed=True,
            specialist_referral=None
        )

# Enhanced Health Bot with integrated CNN, Whisper, Translation and TTS
class AdvancedHealthBot:
    def __init__(self, openai_key=None, gemini_key=None):
        self.ai_analyzer = AIHealthAnalyzer(openai_key, gemini_key)
        self.image_analyzer = CNNMedicalImageAnalyzer()
        self.speech_recognizer = WhisperSpeechRecognizer()
        self.translator = TranslationManager()
        self.tts_manager = TextToSpeechManager()
        
    def analyze_symptoms(self, user_input: str, patient: PatientProfile, 
                         conversation_history: List = None, 
                         uploaded_image: Image.Image = None) -> Dict[str, any]:
        """Comprehensive symptom analysis with CNN image integration"""
        
        image_analysis = None
        combined_input = user_input
        
        # First, analyze the image if provided using CNN
        if uploaded_image:
            image_analysis = self.image_analyzer.analyze_medical_image(uploaded_image)
            if image_analysis and 'auto_symptoms' in image_analysis and image_analysis['auto_symptoms']:
                # Automatically append CNN visual symptoms to user input
                visual_symptoms = image_analysis['auto_symptoms']
                combined_input += f"\n\nCNN Image Analysis: {visual_symptoms}"
        
        # Now, analyze the combined text (symptoms + CNN image findings)
        diagnostic_result = self.ai_analyzer.analyze_symptoms_with_ai(
            combined_input, patient, 
            self._extract_previous_responses(conversation_history)
        )
        
        return self._combine_analyses(diagnostic_result, image_analysis)
    
    def _extract_previous_responses(self, conversation_history: List) -> List[str]:
        """Extract previous responses"""
        if not conversation_history:
            return []
        
        responses = []
        for msg in conversation_history:
            if hasattr(msg, 'role') and msg.role == 'user':
                responses.append(msg.content)
        
        return responses[-5:]
    
    def _combine_analyses(self, diagnostic_result: DiagnosticResult, 
                          image_analysis: Optional[Dict] = None) -> Dict[str, any]:
        """Combine text and CNN image analyses"""
        
        result = asdict(diagnostic_result)
        
        if image_analysis and 'error' not in image_analysis:
            result['image_findings'] = image_analysis
        
        return result

# Initialize session state
def init_session_state():
    if 'current_screen' not in st.session_state:
        st.session_state.current_screen = 'welcome'
    if 'patient_profile' not in st.session_state:
        st.session_state.patient_profile = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'diagnostic_session' not in st.session_state:
        st.session_state.diagnostic_session = {}
    if 'pending_questions' not in st.session_state:
        st.session_state.pending_questions = []
    if 'ai_keys_configured' not in st.session_state:
        st.session_state.ai_keys_configured = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'processed_image_hash' not in st.session_state:
        st.session_state.processed_image_hash = None
    if 'selected_ui_language' not in st.session_state:
        st.session_state.selected_ui_language = 'English'
    if 'translator' not in st.session_state:
        st.session_state.translator = TranslationManager()
    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = TextToSpeechManager()

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime.datetime
    diagnostic_data: Optional[Dict] = None

# Enhanced welcome screen with translation
def render_welcome_screen():
    # Language selector for UI
    selected_ui_lang = st.sidebar.selectbox(
        "🌍 UI Language / भाषा चुनें / ভাষা নির্বাচন করুন", 
        list(SUPPORTED_LANGUAGES.keys()),
        key="ui_language_selector",
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_ui_language)
    )
    
    # FIXED: Check for language change and update session state + rerun
    if selected_ui_lang != st.session_state.selected_ui_language:
        st.session_state.selected_ui_language = selected_ui_lang
        st.rerun()
    
    # Get translated UI text
    ui_texts = st.session_state.translator.get_ui_translations(selected_ui_lang)
    
    st.markdown(f"# {ui_texts['title']}")
    st.markdown(f"### {ui_texts['subtitle']}")
    st.markdown(f"*{ui_texts['language_support']}*")
    
    # System Status
    with st.expander(f"🔧 {ui_texts['system_status']}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Whisper:** {'✅ Available' if WHISPER_AVAILABLE else '❌ Not Available'}")
            st.write(f"**PIL/Images:** {'✅ Available' if PIL_AVAILABLE else '❌ Not Available'}")
        with col2:
            st.write(f"**Transformers:** {'✅ Available' if TRANSFORMERS_AVAILABLE else '❌ Not Available'}")
            st.write(f"**Translation:** {'✅ Available' if TRANSLATOR_AVAILABLE else '❌ Not Available'}")
            st.write(f"**Text-to-Speech:** {'✅ Available' if TTS_AVAILABLE else '❌ Not Available'}")
            st.write(f"**PDF Export:** {'✅ Available' if PDF_AVAILABLE else '❌ Not Available'}")
    
    # API Configuration
    with st.expander(f"🔧 {ui_texts['ai_config']}", expanded=False):
        st.markdown("**For enhanced AI analysis, configure API keys:**")
        openai_key = st.text_input("OpenAI API Key", type="password", help="Optional: For advanced symptom analysis")
        gemini_key = st.text_input("Google Gemini API Key", type="password", help="Optional: For additional AI analysis")
        
        if st.button("Configure AI"):
            if openai_key or gemini_key:
                st.session_state.ai_keys = {'openai': openai_key, 'gemini': gemini_key}
                st.session_state.ai_keys_configured = True
                st.success("AI configuration updated!")
            else:
                st.session_state.ai_keys_configured = False
                st.info("Using CNN + fallback analysis system.")
    
    with st.form("enhanced_patient_form"):
        st.markdown(f"#### {ui_texts['patient_info']}")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(ui_texts['name'], placeholder="Enter your name (optional)")
            age = st.number_input(ui_texts['age'], min_value=1, max_value=120, value=30)
            gender = st.selectbox(ui_texts['gender'], [ui_texts.get('male', "Male"), ui_texts.get('female', "Female"), ui_texts.get('other', "Other"), ui_texts.get('prefer_not_to_say', "Prefer not to say")])
        
        with col2:
            # Enhanced language support for India
            language = st.selectbox(ui_texts['preferred_language'], list(SUPPORTED_LANGUAGES.keys()))
            
        st.markdown(f"#### {ui_texts['medical_history']}")
        col3, col4 = st.columns(2)
        
        with col3:
            medical_history = st.text_area(ui_texts['previous_conditions'], 
                                             placeholder="e.g., diabetes, hypertension")
            medications = st.text_area(ui_texts['current_medications'], 
                                         placeholder="List any medications you're taking")
        
        with col4:
            allergies = st.text_area(ui_texts['known_allergies'], 
                                       placeholder="Food, drug, or environmental allergies")
            family_history = st.text_area(ui_texts['family_history'], 
                                            placeholder="Relevant family medical conditions")
        
        st.markdown(f"#### {ui_texts['lifestyle_info']}")
        lifestyle_col1, lifestyle_col2 = st.columns(2)
        
        with lifestyle_col1:
            smoking = st.selectbox(ui_texts['smoking_status'], [ui_texts.get('never', "Never"), ui_texts.get('former', "Former"), ui_texts.get('current', "Current")])
            alcohol = st.selectbox(ui_texts['alcohol_consumption'], [ui_texts.get('never', "Never"), ui_texts.get('occasional', "Occasional"), ui_texts.get('regular', "Regular"), ui_texts.get('heavy', "Heavy")])
        
        with lifestyle_col2:
            exercise = st.selectbox(ui_texts['exercise_frequency'], [ui_texts.get('never', "Never"), ui_texts.get('rarely', "Rarely"), ui_texts.get('weekly', "2-3 times/week"), ui_texts.get('daily', "Daily")])
            stress_level = st.slider(ui_texts['stress_level'], 1, 10, 5)
        
        if st.form_submit_button(ui_texts['start_analysis'], use_container_width=True):
            patient_profile = PatientProfile(
                name=name if name else "Anonymous",
                age=age,
                gender=gender,
                language=language,
                medical_history=[h.strip() for h in medical_history.split(',') if h.strip()] if medical_history else [],
                medications=[m.strip() for m in medications.split(',') if m.strip()] if medications else [],
                allergies=[a.strip() for a in allergies.split(',') if a.strip()] if allergies else [],
                family_history=[f.strip() for f in family_history.split(',') if f.strip()] if family_history else [],
                lifestyle_factors={
                    'smoking': smoking,
                    'alcohol': alcohol,
                    'exercise': exercise,
                    'stress_level': stress_level
                },
                session_id=str(uuid.uuid4())
            )
            st.session_state.patient_profile = patient_profile
            st.session_state.current_screen = 'consultation'
            st.rerun()

def render_consultation_screen():
    # Get translated UI text
    ui_texts = st.session_state.translator.get_ui_translations(st.session_state.selected_ui_language)
    
    st.markdown(f"# {ui_texts['health_consultation']}")
    
    # Display patient info
    if st.session_state.patient_profile:
        with st.expander(ui_texts['patient_profile'], expanded=False):
            profile = st.session_state.patient_profile
            st.write(f"**Patient:** {profile.name} | **Age:** {profile.age} | **Gender:** {profile.gender} | **Language:** {profile.language}")
            if profile.medical_history:
                st.write(f"**Medical History:** {', '.join(profile.medical_history)}")
    
    # Main consultation interface
    col_chat, col_tools = st.columns([2, 1])
    
    with col_chat:
        st.markdown(f"### {ui_texts['ai_analysis_chat']}")
        
        # Display chat messages
        if not st.session_state.chat_messages:
            st.info(ui_texts['hello_message'])
        
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg.role):
                st.write(msg.content)
                
                if hasattr(msg, 'diagnostic_data') and msg.diagnostic_data:
                    render_diagnostic_results(msg.diagnostic_data, ui_texts)
        
        # Chat input
        user_input = st.chat_input(ui_texts['describe_symptoms'])
        if user_input:
            process_symptoms_with_ai(user_input)
    
    with col_tools:
        st.markdown(f"### {ui_texts['enhanced_tools']}")
        
        # Whisper Speech Recognition
        st.markdown(f"**{ui_texts['voice_input']}**")
        selected_language = st.session_state.patient_profile.language if st.session_state.patient_profile else "English"
        
        if MIC_RECORDER_AVAILABLE:
            try:
                audio_data = mic_recorder(
                    start_prompt="🔴 Start Recording",
                    stop_prompt="⏹️ Stop Recording", 
                    just_once=True,
                    key='mic_recorder'
                )
                
                if audio_data:
                    with st.spinner(f"🔄 Processing speech with Whisper in {selected_language}..."):
                        try:
                            # Get API keys if configured
                            bot = AdvancedHealthBot(
                                st.session_state.ai_keys.get('openai') if 'ai_keys' in st.session_state else None,
                                st.session_state.ai_keys.get('gemini') if 'ai_keys' in st.session_state else None
                            )
                            
                            # Use Whisper speech recognition
                            result = bot.speech_recognizer.transcribe_from_mic_recorder(audio_data, selected_language)
                            
                            if result['success']:
                                recognized_text = result['text']
                                language_used = result.get('language', selected_language)
                                st.success(f"📝 Whisper Recognized ({language_used}): {recognized_text}")
                                
                                # Add language context to the symptom description
                                enhanced_input = f"[Whisper voice input in {language_used}]: {recognized_text}"
                                process_symptoms_with_ai(enhanced_input)
                            else:
                                st.error(f"Whisper recognition failed: {result['error']}")
                        
                        except Exception as e:
                            st.error(f"Whisper processing error: {e}")

            except Exception as e:
                st.warning(f"Mic recorder error: {e}")
                manual_voice_input_whisper(selected_language)
        else:
            manual_voice_input_whisper(selected_language)
        
        st.markdown("---")
        
        # Enhanced CNN Image upload with fixed processing
        st.markdown(f"**{ui_texts['image_analysis']}**")
        st.info("Images analyzed using ResNet50 + specialized dermatology models!")
        
        uploaded_file = st.file_uploader(
            "Upload medical image for CNN analysis", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="CNN will automatically extract visual features and medical symptoms from your image."
        )
        
        if uploaded_file is not None:
            if PIL_AVAILABLE:
                image = Image.open(uploaded_file)
                
                # Create a unique identifier for this image to prevent reprocessing
                image_hash = hash(uploaded_file.getvalue())
                
                # Only process if it's a new image
                if st.session_state.processed_image_hash != image_hash:
                    st.image(image, caption="Image uploaded - CNN analysis in progress...", use_container_width=True)
                    
                    # CNN image analysis with visual feedback
                    with st.spinner("🧠 Running CNN analysis on image..."):
                        try:
                            analyzer = CNNMedicalImageAnalyzer()
                            analysis = analyzer.analyze_medical_image(image)
                            
                            if 'auto_symptoms' in analysis and analysis['auto_symptoms']:
                                st.success("✅ CNN successfully extracted medical features!")
                                st.info(f"CNN Analysis: {analysis['auto_symptoms'][:150]}...")
                                
                                # Show CNN analysis details
                                if 'specialized_analysis' in analysis:
                                    specialized = analysis['specialized_analysis']
                                    st.write(f"**🎯 Dermatology Model:** {specialized.get('primary_prediction', 'Unknown')}")
                                    st.write(f"**📊 Confidence:** {specialized.get('confidence', 0):.2f}")
                                
                                # Store for next chat message and mark as processed
                                st.session_state.uploaded_image = image
                                st.session_state.image_analysis = analysis
                                st.session_state.processed_image_hash = image_hash
                                
                                # Automatically process the CNN image symptoms
                                auto_message = f"I have uploaded an image for CNN analysis. {analysis['auto_symptoms']}"
                                process_symptoms_with_ai(auto_message)
                            else:
                                st.warning("CNN analysis completed but no clear medical features detected.")
                                st.session_state.uploaded_image = image
                                st.session_state.processed_image_hash = image_hash
                                
                        except Exception as e:
                            st.error(f"CNN analysis failed: {e}")
                            st.session_state.uploaded_image = image
                            st.session_state.processed_image_hash = image_hash
                else:
                    # Image already processed, just display it
                    st.image(image, caption="Image already analyzed", use_container_width=True)
                    st.info("This image has already been processed. Upload a new image for fresh analysis.")
            else:
                st.error("PIL not available - cannot process images")
        
        st.markdown("---")
        
        # Quick symptom categories
        st.markdown(f"**{ui_texts['quick_symptoms']}**")
        symptom_categories = [
            ui_texts.get('respiratory', "Respiratory"),
            ui_texts.get('heart_chest', "Heart/Chest"),
            ui_texts.get('digestive', "Digestive"),
            ui_texts.get('neurological', "Neurological"),
            ui_texts.get('skin', "Skin"),
            ui_texts.get('pain', "Pain")
        ]
        
        cols = st.columns(3)
        for i, category in enumerate(symptom_categories):
            if cols[i % 3].button(category, use_container_width=True):
                process_symptoms_with_ai(f"I have symptoms related to {category.lower()}")
        
        st.markdown("---")
        
        # Pending questions
        if st.session_state.pending_questions:
            st.markdown(f"**{ui_texts['follow_up_questions']}**")
            for i, question in enumerate(st.session_state.pending_questions[:3]):
                if st.button(f"{textwrap.shorten(question, width=40, placeholder='...')}", use_container_width=True, key=f"q_{i}"):
                    process_symptoms_with_ai(f"In response to '{question}': ")

def manual_voice_input_whisper(language):
    """Enhanced manual voice input using Whisper"""
    st.info(f"🎤 Manual Whisper Voice Input in {language}")
    
    if not WHISPER_AVAILABLE:
        st.error("Whisper not available. Please install: pip install openai-whisper sounddevice soundfile pydub")
        return
        
    with st.form("whisper_voice_form"):
        duration = st.slider("Recording duration (seconds)", 3, 15, 8)
        if st.form_submit_button(f"🎤 Record with Whisper in {language}"):
            try:
                bot = AdvancedHealthBot(
                    st.session_state.ai_keys.get('openai') if 'ai_keys' in st.session_state else None,
                    st.session_state.ai_keys.get('gemini') if 'ai_keys' in st.session_state else None
                )
                
                result = bot.speech_recognizer.transcribe_with_whisper(duration, language)
                
                if result['success']:
                    recognized_text = result['text']
                    language_used = result.get('language', language)
                    st.success(f"📝 Whisper Recognized ({language_used}): {recognized_text}")
                    
                    # Add language context to the symptom description
                    enhanced_input = f"[Whisper voice input in {language_used}]: {recognized_text}"
                    process_symptoms_with_ai(enhanced_input)
                else:
                    st.error(result['error'])
                    st.info("Tips: Speak clearly, ensure microphone access, check system audio settings.")
                        
            except Exception as e:
                st.error(f"Whisper voice input failed: {e}")

def process_symptoms_with_ai(user_input: str):
    """Process user symptoms using AI analysis with CNN integration and multilingual support"""
    
    if not st.session_state.patient_profile:
        st.error("Please complete patient profile first")
        return

    # Retrieve and clear the uploaded image from session state
    uploaded_image = st.session_state.get('uploaded_image', None)
    if uploaded_image:
        st.session_state.uploaded_image = None
    
    # Add user message
    user_msg = ChatMessage(
        role='user',
        content=user_input,
        timestamp=datetime.datetime.now()
    )
    st.session_state.chat_messages.append(user_msg)
    
    # Initialize health bot with CNN and analyze
    with st.spinner("🧠 CNN + AI analyzing your information..."):
        try:
            # Get API keys if configured
            openai_key = st.session_state.ai_keys.get('openai') if 'ai_keys' in st.session_state else None
            gemini_key = st.session_state.ai_keys.get('gemini') if 'ai_keys' in st.session_state else None
            
            bot = AdvancedHealthBot(openai_key, gemini_key)
            analysis = bot.analyze_symptoms(
                user_input,
                st.session_state.patient_profile,
                st.session_state.chat_messages,
                uploaded_image
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            # Create fallback analysis
            fallback_analyzer = AIHealthAnalyzer()
            analysis = asdict(fallback_analyzer._fallback_analysis(user_input, st.session_state.patient_profile))
    
    # Generate response based on analysis with translation support
    response_content = generate_ai_response_with_translation(analysis)
    
    # Add bot response with diagnostic data
    bot_msg = ChatMessage(
        role='assistant',
        content=response_content,
        timestamp=datetime.datetime.now(),
        diagnostic_data=analysis
    )
    st.session_state.chat_messages.append(bot_msg)
    
    # Update pending questions
    st.session_state.pending_questions = analysis.get('recommended_questions', [])
    
    st.rerun()

def generate_ai_response_with_translation(analysis: Dict) -> str:
    """Generate human-readable response from AI analysis with translation support"""
    
    patient_language = st.session_state.patient_profile.language if st.session_state.patient_profile else "English"
    translator = st.session_state.translator
    
    # Generate response in English first
    response = "Based on your symptoms and profile, here's my comprehensive analysis:\n\n"
    
    # CNN image analysis summary
    if 'image_findings' in analysis:
        image_data = analysis['image_findings']
        
        if 'specialized_analysis' in image_data:
            specialized = image_data['specialized_analysis']
            prediction = specialized.get('primary_prediction', 'Unknown')
            confidence = specialized.get('confidence', 0)
            response += f"**🧠 CNN Dermatology Analysis:** {prediction} (confidence: {confidence:.2f})\n"
            
            if 'top_predictions' in specialized and len(specialized['top_predictions']) > 1:
                other_conditions = [pred['condition'] for pred in specialized['top_predictions'][1:3]]
                response += f"Other possibilities: {', '.join(other_conditions)}\n\n"
        
        elif 'cnn_features' in image_data:
            features = image_data['cnn_features']
            if 'visual_characteristics' in features:
                characteristics = features['visual_characteristics']
                response += "**🧠 CNN Visual Analysis:**\n"
                for aspect, description in characteristics.items():
                    response += f"• {description.capitalize()}\n"
                response += "\n"
        
        quality = image_data.get('image_quality', {}).get('assessment', 'analyzed')
        response += f"*Image quality: {quality}*\n\n"

    # Possible conditions
    if analysis.get('possible_conditions'):
        response += "**🎯 Possible Conditions:**\n"
        for condition in analysis['possible_conditions'][:3]:  # Top 3
            confidence = condition.get('confidence', 0.5) * 100
            response += f"• **{condition['name']}** ({confidence:.0f}% likelihood)\n"
            if condition.get('description'):
                response += f"  *{condition['description'][:100]}...*\n"
    
    # Immediate actions
    if analysis.get('immediate_actions'):
        response += "\n**⚡ Immediate Recommendations:**\n"
        for action in analysis['immediate_actions']:
            response += f"• {action}\n"
    
    # Questions for better diagnosis
    if analysis.get('recommended_questions'):
        response += "\n**❓ To help provide better guidance, could you answer:**\n"
        for i, question in enumerate(analysis['recommended_questions'][:3], 1):
            response += f"{i}. {question}\n"
    
    response += "\n**⚠️ Important:** This analysis incorporates automated CNN image analysis but is for informational purposes only. Always consult a qualified healthcare professional for proper diagnosis and treatment."
    
    # Translate to patient's language if not English
    if patient_language != "English" and TRANSLATOR_AVAILABLE:
        try:
            lang_code = SUPPORTED_LANGUAGES.get(patient_language, {}).get("code", "en")
            if lang_code != "en":
                translated_response = translator.translate_text(response, lang_code)
                # Show both English and translated version
                response = f"**Diagnosis (English):**\n{response}\n\n**Diagnosis ({patient_language}):**\n{translated_response}"
        except Exception as e:
            response += f"\n\n*Translation error: {e}*"
    
    return response

def render_diagnostic_results(diagnostic_data: Dict, ui_texts: Dict):
    """Render detailed diagnostic results including CNN analysis with TTS support"""
    
    patient_language = st.session_state.patient_profile.language if st.session_state.patient_profile else "English"
    
    if 'possible_conditions' in diagnostic_data and diagnostic_data['possible_conditions']:
        with st.expander("🔬 View Detailed Analysis", expanded=False):
            for condition in diagnostic_data['possible_conditions']:
                confidence = condition.get('confidence', 0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{condition['name']}**")
                    if 'description' in condition:
                        st.caption(condition['description'])
                with col2:
                    st.metric("Likelihood", f"{confidence*100:.0f}%")
                    st.progress(confidence)
    
    # Add Text-to-Speech for diagnosis
    if TTS_AVAILABLE and diagnostic_data.get('possible_conditions'):
        with st.expander(f"🔊 {ui_texts.get('play_audio', 'Play Audio Diagnosis')}", expanded=False):
            
            # Generate diagnosis summary for TTS
            diagnosis_summary = "Your diagnosis summary: "
            conditions = diagnostic_data.get('possible_conditions', [])
            if conditions:
                top_condition = conditions[0]
                diagnosis_summary += f"Most likely condition is {top_condition['name']} with {top_condition.get('confidence', 0)*100:.0f}% confidence. "
            
            if diagnostic_data.get('immediate_actions'):
                diagnosis_summary += "Immediate recommendations include: " + ". ".join(diagnostic_data['immediate_actions'][:2]) + ". "
            
            diagnosis_summary += "Please consult a healthcare professional for proper diagnosis and treatment."
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**English Audio:**")
                if st.button("🔊 Play in English", key="tts_english"):
                    try:
                        tts_manager = st.session_state.tts_manager
                        tts_manager.play_diagnosis_audio(diagnosis_summary, "English")
                    except Exception as e:
                        st.error(f"English TTS failed: {e}")
            
            with col2:
                if patient_language != "English":
                    st.write(f"**{patient_language} Audio:**")
                    if st.button(f"🔊 Play in {patient_language}", key=f"tts_{patient_language}"):
                        try:
                            # Translate diagnosis summary first
                            translator = st.session_state.translator
                            lang_code = SUPPORTED_LANGUAGES.get(patient_language, {}).get("code", "en")
                            
                            if TRANSLATOR_AVAILABLE and lang_code != "en":
                                translated_summary = translator.translate_text(diagnosis_summary, lang_code)
                            else:
                                translated_summary = diagnosis_summary
                            
                            tts_manager = st.session_state.tts_manager
                            tts_manager.play_diagnosis_audio(translated_summary, patient_language)
                        except Exception as e:
                            st.error(f"{patient_language} TTS failed: {e}")
    
    if 'image_findings' in diagnostic_data:
        with st.expander("🧠 View CNN Image Analysis Details", expanded=False):
            image_data = diagnostic_data['image_findings']
            
            # Show specialized dermatology analysis if available
            if 'specialized_analysis' in image_data:
                specialized = image_data['specialized_analysis']
                
                st.write("**🎯 Specialized Dermatology Model Results:**")
                st.info(f"Primary Prediction: {specialized.get('primary_prediction', 'Unknown')}")
                st.metric("Confidence", f"{specialized.get('confidence', 0):.3f}")
                
                if 'top_predictions' in specialized:
                    st.write("**Top Predictions:**")
                    for i, pred in enumerate(specialized['top_predictions'][:3], 1):
                        st.write(f"{i}. {pred['condition']} ({pred['probability']:.3f})")
            
            # Show CNN feature analysis
            if 'cnn_features' in image_data:
                features = image_data['cnn_features']
                
                st.write("**🧠 CNN Feature Analysis:**")
                if 'visual_characteristics' in features:
                    characteristics = features['visual_characteristics']
                    for aspect, description in characteristics.items():
                        st.write(f"**{aspect.capitalize()}:** {description}")
                
                if 'feature_vector' in features:
                    st.write("**Feature Vector (first 10 values):**")
                    st.write(features['feature_vector'][:10])
            
            # Show image quality assessment
            if 'image_quality' in image_data:
                quality = image_data['image_quality']
                st.write("**📊 Image Quality Assessment:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Blur Score", quality.get('blur_score', 'N/A'))
                with col2:
                    st.metric("Brightness", quality.get('brightness', 'N/A'))
                with col3:
                    st.metric("Contrast", quality.get('contrast', 'N/A'))
                
                assessment = quality.get('assessment', 'unknown')
                if assessment == 'good':
                    st.success(f"Quality Assessment: {assessment}")
                elif assessment == 'fair':
                    st.warning(f"Quality Assessment: {assessment}")
                else:
                    st.error(f"Quality Assessment: {assessment}")

def render_summary_screen():
    """Enhanced summary screen with CNN analysis summary and multilingual support"""
    
    # Get translated UI text
    ui_texts = st.session_state.translator.get_ui_translations(st.session_state.selected_ui_language)
    
    st.markdown("# 📊 Comprehensive Health Analysis Summary")
    st.markdown("*Including CNN Image Analysis & Translation Results*")
    
    if not st.session_state.patient_profile:
        st.warning("No consultation data available.")
        if st.button("🏠 Start New Consultation"):
            st.session_state.current_screen = 'welcome'
            st.rerun()
        return
    
    profile = st.session_state.patient_profile
    
    # Patient Information Card
    with st.container(border=True):
        st.markdown("### 👤 Patient Profile")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Name:** {profile.name}")
            st.write(f"**Age:** {profile.age}")
        with col2:
            st.write(f"**Gender:** {profile.gender}")
            st.write(f"**Language:** {profile.language}")
        with col3:
             if profile.medical_history:
                st.write(f"**History:** {', '.join(profile.medical_history)}")
             if profile.allergies:
                st.write(f"**Allergies:** {', '.join(profile.allergies)}")
    
    # Technology Summary
    cnn_analyses = 0
    whisper_inputs = 0
    translation_used = profile.language != "English"
    
    for msg in st.session_state.chat_messages:
        if 'CNN' in msg.content or 'cnn' in msg.content.lower():
            cnn_analyses += 1
        if 'Whisper' in msg.content or 'whisper' in msg.content.lower():
            whisper_inputs += 1
    
    with st.container(border=True):
        st.markdown("### 🤖 AI Technology Usage")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CNN Image Analyses", cnn_analyses)
        with col2:
            st.metric("Whisper Voice Inputs", whisper_inputs)
        with col3:
            st.metric("Translation Used", "Yes" if translation_used else "No")
        with col4:
            st.metric("Total AI Interactions", len(st.session_state.chat_messages))
    
    # Consultation Summary
    all_conditions = []
    all_recommendations = set()
    
    for msg in st.session_state.chat_messages:
        if hasattr(msg, 'diagnostic_data') and msg.diagnostic_data:
            data = msg.diagnostic_data
            if 'possible_conditions' in data:
                all_conditions.extend(data['possible_conditions'])
            if 'immediate_actions' in data:
                all_recommendations.update(data['immediate_actions'])
    
    col_cond, col_rec = st.columns(2)

    with col_cond:
        if all_conditions:
            st.markdown("### 🎯 Summary of Possible Conditions")
            
            condition_scores = {}
            for condition in all_conditions:
                name = condition['name']
                confidence = condition.get('confidence', 0)
                condition_scores[name] = max(condition_scores.get(name, 0), confidence)
            
            sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (condition_name, confidence) in enumerate(sorted_conditions[:5], 1):
                st.write(f"{i}. **{condition_name}** ({confidence*100:.0f}% likelihood)")
    
    with col_rec:
        if all_recommendations:
            st.markdown("### 💡 Consolidated Recommendations")
            for rec in list(all_recommendations):
                st.write(f"• {rec}")
    
    # Export and actions
    st.markdown("---")
    st.markdown("### 📄 Next Steps")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏥 Find Nearby Doctors", use_container_width=True):
            st.info("Doctor search functionality would integrate with local medical directories.")
    
    with col2:
        if st.button("📱 Generate QR Summary", use_container_width=True):
            if PIL_AVAILABLE:
                try:
                    qr_data = generate_consultation_summary()
                    qr_img = generate_qr_code(qr_data)
                    
                    if qr_img:
                        # Display QR code
                        st.image(qr_img, width=200, caption="Scan to access summary")
                        
                        # Show what's in the QR code
                        st.json(qr_data)
                    
                except Exception as e:
                    st.error(f"QR generation failed: {e}")
            else:
                st.error("QR code generation requires PIL package.")
    
    with col3:
        # PDF Export Button
        if st.button(ui_texts.get('export_pdf', '📄 Export PDF Summary'), use_container_width=True):
            if PDF_AVAILABLE:
                try:
                    with st.spinner(ui_texts.get('generating_pdf', 'Generating PDF report...')):
                        # Initialize PDF generator
                        pdf_generator = PDFReportGenerator(st.session_state.translator)
                        
                        # Generate comprehensive diagnostic data
                        diagnostic_data = {
                            'possible_conditions': all_conditions,
                            'immediate_actions': list(all_recommendations)
                        }
                        
                        # Generate PDF
                        pdf_buffer = pdf_generator.generate_health_report(
                            st.session_state.patient_profile,
                            st.session_state.chat_messages,
                            diagnostic_data,
                            st.session_state.selected_ui_language
                        )
                        
                        # Create filename
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"health_report_{st.session_state.patient_profile.name}_{timestamp}.pdf"
                        
                        # Provide download
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success(ui_texts.get('pdf_generated', 'PDF report generated successfully!'))
                        
                except Exception as e:
                    st.error(f"{ui_texts.get('pdf_error', 'Error generating PDF report')}: {e}")
                    if "reportlab" in str(e).lower():
                        st.info("To enable PDF export, install: pip install reportlab")
            else:
                st.error("PDF generation requires ReportLab. Install with: pip install reportlab")
    
    with col4:
        if st.button("🔄 New Consultation", use_container_width=True):
            # Reset for new consultation
            for key in ['chat_messages', 'pending_questions', 'diagnostic_session', 'patient_profile', 'uploaded_image']:
                if key in st.session_state:
                    st.session_state[key] = None if key == 'patient_profile' else []
            st.session_state.current_screen = 'welcome'
            st.rerun()

def generate_consultation_summary():
    """Generate structured consultation summary including CNN, Whisper, and Translation usage"""
    
    # Count conditions and recommendations
    all_conditions = []
    all_recommendations = set()
    cnn_analyses = 0
    whisper_inputs = 0
    
    for msg in st.session_state.chat_messages:
        if hasattr(msg, 'diagnostic_data') and msg.diagnostic_data:
            data = msg.diagnostic_data
            if 'possible_conditions' in data:
                all_conditions.extend([c['name'] for c in data['possible_conditions']])
            if 'immediate_actions' in data:
                all_recommendations.update(data['immediate_actions'])
        
        # Count technology usage
        if 'CNN' in msg.content or 'cnn' in msg.content.lower():
            cnn_analyses += 1
        if 'Whisper' in msg.content or 'whisper' in msg.content.lower():
            whisper_inputs += 1
    
    summary_data = {
        'patient_id': st.session_state.patient_profile.session_id,
        'consultation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'patient_name': st.session_state.patient_profile.name,
        'age': st.session_state.patient_profile.age,
        'patient_language': st.session_state.patient_profile.language,
        'ui_language': st.session_state.selected_ui_language,
        'total_messages': len(st.session_state.chat_messages),
        'conditions_analyzed': len(set(all_conditions)),
        'top_conditions': list(set(all_conditions))[:3],
        'recommendations_count': len(all_recommendations),
        'cnn_analyses': cnn_analyses,
        'whisper_inputs': whisper_inputs,
        'translation_used': st.session_state.patient_profile.language != "English",
        'ai_technologies_used': ['CNN Image Analysis', 'Whisper Speech Recognition', 'Google Translation', 'Text-to-Speech'],
        'session_duration_minutes': (datetime.datetime.now() - st.session_state.chat_messages[0].timestamp).seconds // 60 if st.session_state.chat_messages else 0
    }
    
    return summary_data

def generate_qr_code(data):
    """Generate QR code from data"""
    try:
        if not PIL_AVAILABLE:
            return None
            
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(json.dumps(data, indent=2))
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to format suitable for Streamlit
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"QR code generation error: {e}")
        return None

def render_sidebar():
    """Enhanced sidebar with CNN, Whisper, Translation and TTS status"""
    
    with st.sidebar:
        st.markdown("### 🤖 AI Health Assistant")
        st.markdown("*CNN + Whisper + Translation Enhanced*")
        
        # FIXED: Language selector for UI (also in sidebar) with proper change detection
        st.markdown("---")
        st.markdown("#### 🌍 Interface Language")
        
        # Get current index for proper default selection
        current_lang = st.session_state.selected_ui_language
        current_index = list(SUPPORTED_LANGUAGES.keys()).index(current_lang) if current_lang in SUPPORTED_LANGUAGES else 0
        
        selected_ui_lang = st.selectbox(
            "Choose UI Language", 
            list(SUPPORTED_LANGUAGES.keys()),
            key="sidebar_ui_language",
            index=current_index
        )
        
        # FIXED: Check for language change and trigger rerun
        if selected_ui_lang != st.session_state.selected_ui_language:
            st.session_state.selected_ui_language = selected_ui_lang
            st.rerun()
        
        # AI Status indicators
        st.markdown("#### 🔄 System Status")
        
        ai_configured = st.session_state.get('ai_keys_configured', False)
        
        # Check CNN availability
        try:
            cnn_analyzer = CNNMedicalImageAnalyzer()
            cnn_status = "🟢 Ready" if cnn_analyzer.available else "🔴 Failed"
        except:
            cnn_status = "🔴 Failed"
        
        # Check Whisper availability
        try:
            whisper_recognizer = WhisperSpeechRecognizer()
            whisper_status = "🟢 Ready" if whisper_recognizer.available else "🔴 Failed"
        except:
            whisper_status = "🔴 Failed"
        
        services = {
            "Core Analysis Engine": "🟢 Active",
            "CNN Image Analysis": cnn_status,
            "Whisper Speech Recognition": whisper_status,
            "Translation Service": "🟢 Ready" if TRANSLATOR_AVAILABLE else "🔴 Failed",
            "Text-to-Speech": "🟢 Ready" if TTS_AVAILABLE else "🔴 Failed",
            "PDF Export": "🟢 Ready" if PDF_AVAILABLE else "🔴 Failed",
            "Enhanced AI": "🟢 Enabled" if ai_configured else "🟡 Basic Mode"
        }
        
        for service, status in services.items():
            st.write(f"**{service}:** {status}")
        
        # Model information
        with st.expander("🧠 AI Models Info", expanded=False):
            st.write("**CNN Models:**")
            st.write("• ResNet50 (feature extraction)")
            if TRANSFORMERS_AVAILABLE:
                st.write("• Dermatology classifier (specialized)")
            st.write("**Speech Model:**")
            st.write("• Whisper base (multilingual)")
            st.write("**Translation:**")
            st.write("• Google Translate API")
            st.write("**Text-to-Speech:**")
            st.write("• Google TTS (gTTS)")
            st.write("**PDF Generation:**")
            st.write("• ReportLab (multilingual reports)")
        
        st.markdown("---")
        
        # Navigation
        if st.button("🏠 New Consultation", use_container_width=True):
            # Reset session but keep AI configuration and language settings
            for key in ['chat_messages', 'pending_questions', 'diagnostic_session', 'patient_profile', 'uploaded_image']:
                if key in st.session_state:
                    st.session_state[key] = None if key == 'patient_profile' else []
            st.session_state.current_screen = 'welcome'
            st.rerun()
        
        if st.button("📊 View Summary", use_container_width=True):
            if st.session_state.patient_profile and st.session_state.chat_messages:
                st.session_state.current_screen = 'summary'
                st.rerun()
            else:
                st.warning("No consultation data to summarize.")
        
        st.markdown("---")
        
        # Enhanced session statistics
        if st.session_state.chat_messages:
            st.markdown("#### 📈 Session Stats")
            total_messages = len(st.session_state.chat_messages)
            user_messages = len([m for m in st.session_state.chat_messages if m.role == 'user'])
            
            # Count technology usage
            cnn_count = sum(1 for msg in st.session_state.chat_messages if 'CNN' in msg.content or 'cnn' in msg.content.lower())
            whisper_count = sum(1 for msg in st.session_state.chat_messages if 'Whisper' in msg.content or 'whisper' in msg.content.lower())
            
            # Check if translation was used
            translation_used = st.session_state.patient_profile.language != "English" if st.session_state.patient_profile else False
            
            st.write(f"**Total Messages:** {total_messages}")
            st.write(f"**Symptoms Analyzed:** {user_messages}")
            st.write(f"**CNN Analyses:** {cnn_count}")
            st.write(f"**Whisper Inputs:** {whisper_count}")
            st.write(f"**Translation Used:** {'Yes' if translation_used else 'No'}")
            
            start_time = st.session_state.chat_messages[0].timestamp
            duration = datetime.datetime.now() - start_time
            minutes = duration.seconds // 60
            st.write(f"**Session Duration:** {minutes} min")
        
        # Installation requirements
        with st.expander("📦 Requirements", expanded=False):
            st.markdown("""
            **Required packages:**
            ```
            pip install torch torchvision
            pip install openai-whisper
            pip install sounddevice soundfile pydub
            pip install opencv-python
            pip install deep-translator
            pip install gtts
            pip install reportlab
            pip install transformers (optional)
            pip install streamlit-mic-recorder (optional)
            pip install qrcode[pil] (optional)
            ```
            """)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer**")
        st.caption("This tool uses CNN, Whisper AI, translation services, and PDF generation but provides informational analysis only. Not a substitute for professional medical advice, diagnosis, or treatment.")

# Main Application
def main():
    init_session_state()
    render_sidebar()
    
    # Route to appropriate screen
    try:
        if st.session_state.current_screen == 'welcome':
            render_welcome_screen()
        elif st.session_state.current_screen == 'consultation':
            render_consultation_screen()
        elif st.session_state.current_screen == 'summary':
            render_summary_screen()
        else:
            st.session_state.current_screen = 'welcome'
            st.rerun()
    except Exception as e:
        st.error(f"An application error occurred: {e}")
        st.info("Please click the button below to restart the application.")
        if st.button("🔄 Restart Application"):
             # Full reset
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()