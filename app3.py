
import streamlit as st
import sqlite3
import re
import hashlib
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from textblob import TextBlob
from transformers import pipeline
import torch
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import requests
import base64
from PIL import Image
import io
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.card import card
import time
import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import docx
import PyPDF2
import os
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64
import sqlite3
import textstat

def add_models_table():
    """Adds the models table to the database if it doesn't exist."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # Corrected the column name from 'is_active' to 'status' and changed the data type
    cur.execute("""CREATE TABLE IF NOT EXISTS models (
                  model_id TEXT PRIMARY KEY,
                  display_name TEXT,
                  remarks TEXT,
                  status TEXT DEFAULT 'active',
                  task_type TEXT NOT NULL DEFAULT 'summarization')""") # Added task_type
    conn.commit()
    conn.close()

add_models_table()
print("Checked/Created 'models' table in users.db")

# -----------------------------------------------
# HUGGING FACE - AI Model Loading and Functions
# -----------------------------------------------
device = 0 if torch.cuda.is_available() else -1


def load_summarizer(model_id="google/pegasus-cnn_dailymail"):
    """Loads the summarization model and tokenizer from Hugging Face."""
    try:
        summarizer_pipeline = pipeline("summarization", model=model_id, device=device)
        tokenizer = summarizer_pipeline.tokenizer
        return summarizer_pipeline, tokenizer
    except Exception as e:
        st.error(f"Failed to load summarization model {model_id}: {e}")
        return None, None


def load_paraphraser(model_id="humarin/chatgpt_paraphraser_on_T5_base"):
    """Loads the paraphrasing model and tokenizer from Hugging Face."""
    try:
        paraphraser_pipeline = pipeline("text2text-generation", model=model_id, device=device)
        tokenizer = paraphraser_pipeline.tokenizer
        return paraphraser_pipeline, tokenizer
    except Exception as e:
        st.error(f"Failed to load paraphrasing model {model_id}: {e}")
        return None, None


def load_qa_model(model_id="distilbert-base-cased-distilled-squad"):
    """Loads a Question Answering model."""
    try:
        qa_pipeline = pipeline("question-answering", model=model_id, device=device)
        return qa_pipeline
    except Exception as e:
        st.error(f"Failed to load Q&A model {model_id}: {e}")
        return None


def load_text_generation_model(model_id="gpt2"):
    """Loads a Text Generation model."""
    try:
        text_gen_pipeline = pipeline("text-generation", model=model_id, device=device)
        return text_gen_pipeline
    except Exception as e:
        st.error(f"Failed to load Text Generation model {model_id}: {e}")
        return None


def answer_question(qa_pipeline, context, question):
    """Generates an answer from a context."""
    if not qa_pipeline:
        return "Q&A model not loaded."
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"Q&A failed: {e}")
        return "Error generating answer."


def generate_text(text_gen_pipeline, prompt, max_length=50):
    """Generates text from a prompt."""
    if not text_gen_pipeline:
        return "Text Generation model not loaded."
    try:
        result = text_gen_pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Text generation failed: {e}")
        return "Error generating text."

def summarize_text(summarizer_pipeline, tokenizer, text, min_length=30, max_length=150):
    """Generates a summary for the given text using the model's tokenizer."""
    if not summarizer_pipeline or not tokenizer:
        return "Summarization model not loaded."

    max_model_input_length = 1000

    if len(text) > max_model_input_length:
        st.warning(f"Input text is too long ({len(text)} characters). Truncating to {max_model_input_length} characters for summarization.")
        text = text[:max_model_input_length]

    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt", padding=True)

    if device != -1:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    try:
        summary_ids = summarizer_pipeline.model.generate(inputs["input_ids"], num_beams=5, max_length=max_length, min_length=min_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return "Error generating summary."


def paraphrase_text(paraphraser_pipeline, tokenizer, text, num_return_sequences=3):
    """Generates paraphrased versions of the given text using the model's tokenizer."""
    if not paraphraser_pipeline or not tokenizer:
        return ["Paraphrasing model not loaded."]

    max_model_input_length = 1000

    if len(text) > max_model_input_length:
         st.warning(f"Input text is too long ({len(text)} characters). Truncating to {max_model_input_length} characters for paraphrasing.")
         text = text[:max_model_input_length]

    inputs = tokenizer(f"paraphrase: {text}", max_length=1024, truncation=True, return_tensors="pt", padding=True)

    if device != -1:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    try:
        paraphrased_ids = paraphraser_pipeline.model.generate(
            inputs["input_ids"],
            num_beams=5,
            num_return_sequences=num_return_sequences,
            max_length=128,
            early_stopping=True
        )
        paraphrased_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in paraphrased_ids]
        return paraphrased_texts
    except Exception as e:
        st.error(f"Paraphrasing failed: {e}")
        return ["Error generating paraphrases."]

# -----------------------------------------------
# VOICE AND LANGUAGE FUNCTIONS
# -----------------------------------------------

def audio_bytes_to_text(audio_data):
    if not audio_data or not isinstance(audio_data, dict) or 'bytes' not in audio_data:
        st.warning("No valid audio data received from microphone.")
        return ""

    audio_bytes = audio_data['bytes']
    sample_rate = audio_data.get('sample_rate', 16000)
    sample_width = audio_data.get('sample_width', 2)

    if not audio_bytes:
        st.warning("No audio bytes found in the recorded data.")
        return ""

    try:
        r = sr.Recognizer()
        audio_data_sr = sr.AudioData(audio_bytes, sample_rate, sample_width)

        text = r.recognize_google(audio_data_sr)
        return text
    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        st.error(f"An error occurred during speech processing: {e}")
        return ""


def text_to_speech(text):
    try:
        engine = pyttsx3.init()

        # Get available voices
        voices = engine.getProperty('voices')

        if not voices:
            st.warning("No text-to-speech voices found on this system.")
            return

        # Try to find an English voice
        selected_voice = None
        for voice in voices:
            # Check if voice has languages attribute and contains 'en'
            if hasattr(voice, 'languages') and voice.languages:
                if any('en' in lang for lang in voice.languages):
                    selected_voice = voice
                    break
            # Fallback: check voice ID for English indicators
            elif hasattr(voice, 'id') and ('en' in voice.id.lower() or 'english' in voice.id.lower()):
                selected_voice = voice
                break

        # If no English voice found, use the first available voice
        if selected_voice is None and voices:
            selected_voice = voices[0]
            st.info(f"Using available voice: {selected_voice.id}")

        if selected_voice:
            try:
                engine.setProperty('voice', selected_voice.id)
            except Exception as voice_error:
                st.warning(f"Could not set voice '{selected_voice.id}': {voice_error}")
                # Continue with default voice

        # Set speech properties
        engine.setProperty('rate', 150)  # Speed percent
        engine.setProperty('volume', 0.9)  # Volume 0-1

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")
        st.info("Text-to-speech might not be fully supported in this environment. You may need to install additional voice packages on your system.")

def detect_language(text):
    try:
        translator = Translator()
        detection = translator.detect(text)
        return detection.lang
    except:
        return "en"

def translate_text(text, dest_language="en"):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except:
        return text

# -----------------------------------------------
# FILE PROCESSING FUNCTIONS
# -----------------------------------------------

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

# -----------------------------------------------
# DATABASE AND USER MANAGEMENT
# -----------------------------------------------

def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # Updated 'users' table with new preference columns
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                     username TEXT PRIMARY KEY, name TEXT, email TEXT UNIQUE,
                     age_category TEXT, language TEXT DEFAULT 'English',
                     profile_pic BLOB, password TEXT, theme TEXT DEFAULT 'light',
                     is_active INTEGER DEFAULT 1, default_model TEXT,
                     reading_preferences TEXT, content_type TEXT)""")

    # Updated 'submissions' table with new readability score columns
    cur.execute("""CREATE TABLE IF NOT EXISTS submissions (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT,
                     timestamp TEXT,
                     model_id_used TEXT,
                     task_type TEXT,
                     input_text TEXT,
                     output_text TEXT,
                     sentiment TEXT,
                     flesch_reading_ease REAL,
                     flesch_kincaid_grade REAL,
                     gunning_fog REAL,
                     smog_index REAL,
                     FOREIGN KEY(username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS models (
                     model_id TEXT PRIMARY KEY,
                     display_name TEXT,
                     remarks TEXT,
                     status TEXT DEFAULT 'active',
                     task_type TEXT NOT NULL DEFAULT 'summarization')""")
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

init_db()

# Insert a test user and default models
def insert_initial_data():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    # Insert test user
    cur.execute("SELECT * FROM users WHERE username=?", ("test",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username, name, email, age_category, language, password, theme, default_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("test", "Test User", "test@test.com", "15 - 20", "English", hash_password("test"), "light", "google/pegasus-cnn_dailymail")) # Set a default model
        conn.commit()

    # Insert default models if not exist
    default_models = [
        ("google/pegasus-cnn_dailymail", "Pegasus CNN/DailyMail (Summarization)", "Good for news articles"),
        ("humarin/chatgpt_paraphraser_on_T5_base", "ChatGPT Paraphraser (Paraphrasing)", "Effective for rephrasing text"),
    ]
    for model_id, display_name, remarks in default_models:
        cur.execute("SELECT * FROM models WHERE model_id=?", (model_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO models (model_id, display_name, remarks, status) VALUES (?, ?, ?, ?)",
                        (model_id, display_name, remarks, 'active'))
            conn.commit()

    conn.close()

insert_initial_data()


def valid_username(username):
    return re.match(r"^[A-Za-z][A-Za-z0-9_]*$", username)

def valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def export_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for i, row in df.iterrows():
        # Safely access all the columns from the full DataFrame
        ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
        task = row.get('task_type', 'N/A')
        sentiment = row.get('sentiment', 'N/A')
        grade = row.get('flesch_kincaid_grade', 0.0)

        input_text = row['input_text'][:200] + '...' if len(row['input_text']) > 200 else row['input_text']
        output_text = row['output_text'][:200] + '...' if len(row['output_text']) > 200 else row['output_text']

        # Create the text block for the PDF entry
        text_content = (
            f"Timestamp: {ts} | Task: {task}\n"
            f"Sentiment: {sentiment} | Readability Grade: {grade:.2f}\n"
            f"------------------------------------------------------------------\n"
            f"Input Text:\n{input_text}\n\n"
            f"Output Text:\n{output_text}"
        )

        pdf.multi_cell(0, 6, text_content.encode('latin-1', 'replace').decode('latin-1'), border=1)
        pdf.cell(0, 5, ln=True) # Add a space between entries

    return pdf.output(dest="S").encode("latin-1")

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: return "Positive"
    elif polarity < -0.1: return "Negative"
    else: return "Neutral"

def image_to_binary(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def binary_to_image(binary_data):
    return Image.open(io.BytesIO(binary_data))

def get_user_theme(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT theme FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else "light"

def update_user_theme(username, theme):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET theme=? WHERE username=?", (theme, username))
    conn.commit()
    conn.close()

def is_user_active(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT is_active FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else 1

def toggle_user_status(username, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_active=? WHERE username=?", (status, username))
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM submissions WHERE username=?", (username,))
    cur.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

def update_password(username, new_password):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET password=? WHERE username=?", (hash_password(new_password), username))
    conn.commit()
    conn.close()

def get_active_models():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT model_id, display_name, remarks FROM models WHERE status='active'", conn)
    conn.close()
    return df

def get_all_models():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT model_id, display_name, remarks, status FROM models", conn)
    conn.close()
    return df

def add_model(model_id, display_name, remarks, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO models (model_id, display_name, remarks, status) VALUES (?, ?, ?, ?)",
                    (model_id, display_name, remarks, status))
        conn.commit()
        st.success(f"Model '{display_name}' added successfully.")
    except sqlite3.IntegrityError:
        st.error(f"Model ID '{model_id}' already exists.")
    conn.close()

def update_model(model_id, display_name, remarks, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE models SET display_name=?, remarks=?, status=? WHERE model_id=?",
                (display_name, remarks, status, model_id))
    conn.commit()
    st.success(f"Model '{display_name}' updated successfully.")
    conn.close()

def delete_model(model_id):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE model_id=?", (model_id,))
    conn.commit()
    st.success(f"Model '{model_id}' deleted successfully.")
    conn.close()

# -----------------------------------------------
# UI ENHANCEMENTS
# -----------------------------------------------

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_welcome = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_vyLwnL.json")
lottie_login = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kcs1arba.json")
lottie_analytics = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vybwn7df.json")
lottie_voice = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ttvteyse.json")
lottie_models = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yisb3l0x.json")

def custom_header():
    """Creates a custom, consistent header for all pages."""
    # Ensure user is logged in to fetch data
    if st.session_state.user:
        conn = sqlite3.connect("users.db")
        user_data = pd.read_sql_query("SELECT name, profile_pic FROM users WHERE username=?", conn, params=(st.session_state.user,))
        conn.close()

        user_name = user_data.iloc[0]['name']
        profile_pic_binary = user_data.iloc[0]['profile_pic']

        # Custom CSS for the header
        st.markdown("""
            <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 1rem;
                border-bottom: 1px solid #333;
                width: 100%;
            }
            .header-left {
                font-size: 1.5rem;
                font-weight: bold;
            }
            .header-right {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            .profile-avatar-header {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                object-fit: cover;
                border: 2px solid #555;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header Layout
        title_col, nav_col = st.columns([3, 2])

        with title_col:
            st.markdown(f'<span class="header-left">Welcome back, {user_name}!</span>', unsafe_allow_html=True)

        with nav_col:
            cols = st.columns([1, 1, 1, 0.5])
            with cols[0]:
                if st.button("Dashboard", use_container_width=True):
                    st.session_state.page = "dashboard"
                    st.rerun()
            with cols[1]:
                if st.session_state.theme == "light":
                    if st.button("🌙 Dark", use_container_width=True):
                        st.session_state.theme = "dark"
                        update_user_theme(st.session_state.user, "dark")
                        st.rerun()
                else:
                    if st.button("☀️ Light", use_container_width=True):
                        st.session_state.theme = "light"
                        update_user_theme(st.session_state.user, "light")
                        st.rerun()
            with cols[2]:
                if st.button("Logout", use_container_width=True):
                    st.session_state.user = None
                    st.session_state.page = "login"
                    st.rerun()
            with cols[3]:
                 if profile_pic_binary:
                    image_b64 = base64.b64encode(profile_pic_binary).decode()
                    st.markdown(f'<img src="data:image/png;base64,{image_b64}" class="profile-avatar-header">', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)


def apply_theme(theme):
    if theme == "dark":
        st.markdown(f"""
        <style>
        .main {{
            background-color: #0E1117;
            color: #FAFAFA;
        }}
        .main-header {{
            font-size: 2.5rem;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #FF4B4B;
            border-bottom: 2px solid #FF4B4B;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }}
        .feature-card {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 18px;
            border: 2px solid #FF4B4B;
            background-color: #FF4B4B;
            color: white;
            padding: 8px 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #0E1117;
            color: #FF4B4B;
        }}
        .secondary-button {{
            background-color: #262730 !important;
            color: #FF4B4B !important;
            border: 2px solid #FF4B4B !important;
        }}
        .secondary-button:hover {{
            background-color: #FF4B4B !important;
            color: white !important;
        }}
        .text-input {{
            border-radius: 10px;
            background-color: #262730;
            color: white;
        }}
        .metric-card {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            text-align: center;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #FF4B4B;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #CCCCCC;
        }}
        .profile-avatar {{
            border-radius: 50%;
            border: 2px solid #FF4B4B;
            box_shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            object-fit: cover;
        }}
        .section-spacing {{
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <style>
        .main {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #31333F;
        }}
        .main-header {{
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }}
        .feature-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 18px;
            border: 2px solid #4CAF50;
            background-color: #4CAF50;
            color: white;
            padding: 8px 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: white;
            color: #4CAF50;
        }}
        .secondary-button {{
            background-color: #f0f2f6 !important;
            color: #1f77b4 !important;
            border: 2px solid #1f77b4 !important;
        }}
        .secondary-button:hover {{
            background-color: #1f77b4 !important;
            color: white !important;
        }}
        .text-input {{
            border-radius: 10px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box_shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #7f8c8d;
        }}
        .profile-avatar {{
            border-radius: 50%;
            border: 2px solid #1f77b4;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            object-fit: cover;
        }}
        .section-spacing {{
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }}
        </style>
        """, unsafe_allow_html=True)

def animate_transition():
    with st.spinner("Loading..."):
        time.sleep(0.5)

# -----------------------------------------------
# STREAMLIT APPLICATION UI
# -----------------------------------------------
st.set_page_config(page_title="TextMorph App", layout="wide", page_icon="📝")

if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "voice_input_text" not in st.session_state:
    st.session_state.voice_input_text = ""
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""
if "generated_paraphrases" not in st.session_state:
    st.session_state.generated_paraphrases = []
if "current_summary_model_id" not in st.session_state: # Changed to store model ID
    st.session_state.current_summary_model_id = None
if "current_paraphrase_model_id" not in st.session_state: # Changed to store model ID
    st.session_state.current_paraphrase_model_id = None
if "summarizer_pipeline" not in st.session_state: # Store loaded pipeline
    st.session_state.summarizer_pipeline = None
if "summarizer_tokenizer" not in st.session_state: # Store loaded tokenizer
    st.session_state.summarizer_tokenizer = None
if "paraphraser_pipeline" not in st.session_state: # Store loaded pipeline
    st.session_state.paraphraser_pipeline = None
if "paraphraser_tokenizer" not in st.session_state: # Store loaded tokenizer
    st.session_state.paraphraser_tokenizer = None


if st.session_state.user:
    user_theme = get_user_theme(st.session_state.user)
    st.session_state.theme = user_theme

apply_theme(st.session_state.theme)

AGE_CATEGORIES = [f"{i} - {i+5}" for i in range(15, 76, 5)]

# ---------- Login Page ----------
if st.session_state.page == "login":
    # NEW: A modern, centered layout
    st.markdown('<h1 class="main-header" style="text-align: center;">TextMorph</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Intelligent Text Transformation at Your Fingertips.</p>', unsafe_allow_html=True)

    # Use columns to center the login form
    _, login_col, _ = st.columns([1, 1.5, 1])

    with login_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        # Display Lottie animation inside the card for visual appeal
        if lottie_login:
            st_lottie(lottie_login, height=200, key="login_lottie")

        st.header("User Login")
        username = st.text_input("Username", help="Enter your username", label_visibility="collapsed", placeholder="Username")
        password = st.text_input("Password", type="password", help="Enter your password", label_visibility="collapsed", placeholder="Password")

        if st.button("Login", key="login_btn", use_container_width=True, type="primary"):
            if (username == "shashi" and password == "26092004") or (username == "pranavi" and password == "20112003"):
                st.session_state.user = username
                st.session_state.page = "admin"
                st.success("Admin login successful")
                animate_transition()
                st.rerun()
            else:
                if not is_user_active(username):
                    st.error("This account has been blocked. Please contact an administrator.")
                else:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                                (username, hash_password(password)))
                    user = cur.fetchone()
                    conn.close()
                    if user:
                        st.session_state.user = username
                        st.session_state.page = "home"
                        st.success("Login successful")
                        animate_transition()
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Secondary action buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register", key="register_btn", use_container_width=True):
                st.session_state.page = "register"
                animate_transition()
                st.rerun()
        with col_b:
            if st.button("Reset Password", key="forgot_btn", use_container_width=True):
                st.session_state.page = "forgot"
                animate_transition()
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Register Page ----------
elif st.session_state.page == "register":
    # NEW: A modern, centered layout consistent with the login page
    st.markdown('<h1 class="main-header" style="text-align: center;">Create an Account</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Join us and start transforming your text.</p>', unsafe_allow_html=True)

    # Use columns to center the registration form
    _, register_col, _ = st.columns([1, 1.8, 1])

    with register_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        # Using a form to group inputs
        with st.form("registration_form"):
            st.subheader("Personal Details")
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email address")
            age_category = st.selectbox("Select Age Category", AGE_CATEGORIES)

            st.markdown("---")
            st.subheader("Account Credentials")
            username = st.text_input("Username", placeholder="Choose a unique username")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")
            verify_password = st.text_input("Verify Password", type="password", placeholder="Re-enter your password")

            st.markdown("---")
            st.subheader("Profile Picture (Optional)")
            profile_pic = st.file_uploader("Upload a profile picture", type=['png', 'jpg', 'jpeg'])

            # Submit button for the form
            submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submitted:
                if not valid_username(username):
                    st.error("Invalid username format. Must start with a letter and contain only letters, numbers, and underscores.")
                elif not valid_email(email):
                    st.error("Invalid email format. Please enter a valid email address.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif password != verify_password:
                    st.error("Passwords do not match")
                else:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    try:
                        profile_pic_binary = None
                        if profile_pic:
                            image = Image.open(profile_pic)
                            profile_pic_binary = image_to_binary(image)

                        active_models = get_active_models()
                        default_model_id = active_models['model_id'].iloc[0] if not active_models.empty else None

                        cur.execute("INSERT INTO users (username, name, email, age_category, language, password, profile_pic, theme, default_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (username, name, email, age_category, "English", hash_password(password), profile_pic_binary, st.session_state.theme, default_model_id))
                        conn.commit()
                        st.success("Account created successfully! Please login.")
                        st.session_state.page = "login"
                        animate_transition()
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Username or Email already exists")
                    finally:
                        conn.close()

        # "Back to Login" button outside the form
        if st.button("⬅️ Back to Login", key="back_login", use_container_width=True):
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

st.set_page_config(page_title="Home", layout="wide")

if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "ai_output" not in st.session_state:
    st.session_state.ai_output = ""
if "__latest_source_text" not in st.session_state:
    st.session_state.__latest_source_text = ""
# ---------- Home Page (AI Tasks + Readability Analysis) ----------
elif st.session_state.page == "home":
    custom_header()

    # ---------- Session State ----------
    st.session_state.setdefault("readability_scores", None)
    st.session_state.setdefault("manual_text", "")
    st.session_state.setdefault("qa_question", "")
    theme = st.session_state.get("theme", "dark")

    # ---------- CSS ----------
    st.markdown("""
    <style>
    .feature-card {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(128,128,128,0.10), rgba(128,128,128,0.05));
        text-align: center;
        transition: transform .2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; }
    .metric-label { font-size: .85rem; opacity: .85; margin-bottom: .5rem; }
    .section-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(128,128,128,0.25), transparent);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0; padding-top: 8px; padding-bottom: 8px;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(255,255,255,0.10); }
    .hint {
        font-size: .85rem; opacity: .8; padding: .35rem .6rem; border-radius: 6px;
        background: rgba(125,125,125,.12); display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- DB: Active Models ----------
    conn = sqlite3.connect("users.db")
    active_models_df = pd.read_sql_query("SELECT * FROM models WHERE status='active'", conn)
    conn.close()

    # ---------- Split Screen ----------
    left, right = st.columns([1.15, 1.85], gap="large")

    # ======================================================================================
    # LEFT: Inputs (Model selection, text input, question, process)
    # ======================================================================================
    with left:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">🤖 Select AI Task & Model</p>', unsafe_allow_html=True)

        selected_task_type = None
        selected_model_id = None
        selected_display_name = None

        if not active_models_df.empty:
            model_options = {
                row['display_name']: {'id': row['model_id'], 'task': row['task_type']}
                for _, row in active_models_df.iterrows()
            }

            colA, colB = st.columns([2, 1])
            with colA:
                selected_display_name = st.selectbox(
                    "Choose an AI Model",
                    list(model_options.keys()),
                    index=0 if len(model_options) > 0 else None,
                    help="Pick the model you want to run on your text"
                )
            with colB:
                selected_model_id = model_options[selected_display_name]['id']
                selected_task_type = model_options[selected_display_name]['task']
                st.info(f"**Task Type:** {selected_task_type.replace('_', ' ').title()}")

        else:
            st.warning("No active models available. Please contact an administrator.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ------------- Input Form -------------
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">📥 Input Your Text</p>', unsafe_allow_html=True)

        output_result = ""
        input_data_for_db = ""
        source_text = ""

        with st.form("home_input_form", clear_on_submit=False):
            tab1, tab2 = st.tabs(["✏️ Manual", "📁 Upload"])
            with tab1:
                source_text = st.text_area(
                    "Enter text",
                    value=st.session_state.get("manual_text", ""),
                    key="manual_text",
                    height=200,
                    placeholder="Paste or type your text here...",
                    label_visibility="collapsed",
                )
                st.caption("Tip: You can paste long content; output will show in the right panel.")

            with tab2:
                uploaded_file = st.file_uploader(
                    "Choose a file (.txt, .pdf, .docx)",
                    type=['txt', 'pdf', 'docx']
                )
                if uploaded_file is not None:
                    with st.spinner("Extracting text..."):
                        if uploaded_file.type == "text/plain":
                            source_text = uploaded_file.getvalue().decode("utf-8")
                        elif uploaded_file.type == "application/pdf":
                            source_text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            source_text = extract_text_from_docx(uploaded_file)
                    if source_text:
                        st.text_area("📝 Extracted Text (preview)", source_text, height=150, disabled=True)
                    else:
                        st.error("Could not extract text from the uploaded file.")

            # Extra field for QA
            if selected_task_type == "question_answering":
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.text_input(
                    "❓ Your Question",
                    key="qa_question",
                    placeholder="What would you like to ask about this text?"
                )
                process_button_label = "❓ Find Answer & Analyze Readability"
            else:
                process_button_label = "✨ Process Text & Analyze Readability"

            # Compact/Detail toggle
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])
            with c1:
                detailed_view = st.toggle("Detailed Charts", value=True, help="Turn off for a compact summary")
            with c2:
                st.markdown('<span class="hint">Press Ctrl/Cmd + Enter to submit</span>', unsafe_allow_html=True)

            # Action buttons
            col_go, col_clear = st.columns([3,1])
            with col_go:
                process_clicked = st.form_submit_button(
                    process_button_label,
                    use_container_width=True,
                    type="primary",
                    disabled=(selected_task_type is None)
                )
            with col_clear:
                clear_clicked = st.form_submit_button("Clear", use_container_width=True)

            if clear_clicked:
                st.session_state.manual_text = ""
                st.session_state.qa_question = ""
                st.session_state.readability_scores = None
                st.toast("Cleared inputs.")
                st.stop()

        st.markdown('</div>', unsafe_allow_html=True)

        # ------------- Processing -------------
        if process_clicked:
            output_result = ""
            st.session_state.readability_scores = None

            if selected_task_type == "question_answering":
                has_text = bool(source_text.strip()) and bool(st.session_state.qa_question.strip())
            else:
                has_text = bool(source_text.strip())

            if not has_text:
                st.warning("Please provide the required text input(s).")
            else:
                with st.status("Running model & analyzing readability…", expanded=False) as status:
                    if selected_task_type == "question_answering":
                        model_pipeline = load_qa_model(selected_model_id)
                        output_result = answer_question(model_pipeline, source_text, st.session_state.qa_question)
                        input_data_for_db = f"Context: {source_text[:500]}...\n\nQuestion: {st.session_state.qa_question}"
                    elif selected_task_type == "summarization":
                        model_pipeline, tokenizer = load_summarizer(selected_model_id)
                        output_result = summarize_text(model_pipeline, tokenizer, source_text)
                        input_data_for_db = source_text
                    elif selected_task_type == "paraphrasing":
                        model_pipeline, tokenizer = load_paraphraser(selected_model_id)
                        paraphrases = paraphrase_text(model_pipeline, tokenizer, source_text)
                        output_result = "\n\n".join([f"Option {i+1}: {p}" for i, p in enumerate(paraphrases)])
                        input_data_for_db = source_text
                    elif selected_task_type == "text_generation":
                        model_pipeline = load_text_generation_model(selected_model_id)
                        output_result = generate_text(model_pipeline, source_text, max_length=150)
                        input_data_for_db = source_text

                    st.session_state.readability_scores = {
                        "Flesch Reading Ease": textstat.flesch_reading_ease(source_text),
                        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(source_text),
                        "Gunning Fog": textstat.gunning_fog(source_text),
                        "SMOG Index": textstat.smog_index(source_text),
                        "Coleman-Liau Index": textstat.coleman_liau_index(source_text),
                        "Automated Readability Index": textstat.automated_readability_index(source_text),
                        "Dale-Chall Readability": textstat.dale_chall_readability_score(source_text),
                        "Linsear Write Formula": textstat.linsear_write_formula(source_text)
                    }

                    status.update(label="Saving to dashboard…")
                    try:
                        conn = sqlite3.connect("users.db")
                        cur = conn.cursor()
                        s = st.session_state.readability_scores
                        sentiment = analyze_sentiment(input_data_for_db)

                        cur.execute("""
                            INSERT INTO submissions
                            (username, timestamp, model_id_used, task_type, input_text, output_text, sentiment,
                             flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            st.session_state.user,
                            str(datetime.datetime.now()),
                            selected_model_id,
                            selected_task_type,
                            input_data_for_db,
                            output_result,
                            sentiment,
                            s['Flesch Reading Ease'],
                            s['Flesch-Kincaid Grade'],
                            s['Gunning Fog'],
                            s['SMOG Index']
                        ))
                        conn.commit()
                        st.toast("✅ Result saved to your dashboard!")
                    except sqlite3.Error as e:
                        st.error(f"Database error: {e}")
                    finally:
                        if conn: conn.close()

                    status.update(label="Done", state="complete")

        if output_result:
            st.session_state["__latest_output_result"] = output_result
        if source_text:
            st.session_state["__latest_source_text"] = source_text
        st.session_state["__latest_detailed_view"] = detailed_view

    # ======================================================================================
    # RIGHT: Results (AI Output, Readability, Visualizations)
    # ======================================================================================
    with right:
        if st.session_state.get("__latest_output_result"):
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">📊 AI Model Output</p>', unsafe_allow_html=True)
            with st.expander("View AI Output", expanded=True):
                st.success(st.session_state["__latest_output_result"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.readability_scores:
            s = st.session_state.readability_scores
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">📈 Readability Analysis</p>', unsafe_allow_html=True)
            st.caption("Analysis of your input text's readability metrics")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                ease = s['Flesch Reading Ease']
                color = "#63d471" if ease > 70 else ("#fbd46d" if ease > 50 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{ease:.1f}</div>
                    <div class="metric-label">Flesch Reading Ease</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Higher = easier to read")
            with m2:
                fk = s['Flesch-Kincaid Grade']
                color = "#63d471" if fk <= 8 else ("#fbd46d" if fk <= 12 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{fk:.1f}</div>
                    <div class="metric-label">Flesch-Kincaid Grade</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("U.S. grade level")
            with m3:
                fog = s['Gunning Fog']
                color = "#63d471" if fog < 10 else ("#fbd46d" if fog < 15 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{fog:.1f}</div>
                    <div class="metric-label">Gunning Fog</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Years of education")
            with m4:
                smog = s['SMOG Index']
                color = "#63d471" if smog < 10 else ("#fbd46d" if smog < 15 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{smog:.1f}</div>
                    <div class="metric-label">SMOG Index</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Years of education")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            detailed_view = st.session_state.get("__latest_detailed_view", True)

            if detailed_view:
                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
                    ["📊 Score Comparison", "📈 Grade Level", "🎯 Radar Chart", "📋 Text Stats"]
                )
                with viz_tab1:
                    st.markdown("#### Readability Scores Comparison")
                    score_names = list(s.keys())[:4]
                    score_values = [s[name] for name in score_names]

                    colors = []
                    for i, name in enumerate(score_names):
                        score = score_values[i]
                        if "Ease" in name:
                            colors.append("#63d471" if score > 70 else ("#fbd46d" if score > 50 else "#f06a6a"))
                        else:
                            colors.append("#63d471" if score < 10 else ("#fbd46d" if score < 15 else "#f06a6a"))

                    fig = go.Figure(data=[go.Bar(
                        x=score_names, y=score_values, marker_color=colors,
                        text=score_values, texttemplate='%{text:.1f}', textposition='auto'
                    )])
                    fig.update_layout(height=400, showlegend=False, yaxis_title="Score", xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("##### Score Interpretation")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Flesch Reading Ease", f"{s['Flesch Reading Ease']:.1f}",
                                  help="90-100: Very Easy, 60-70: Standard, 0-30: Very Confusing")
                    with c2:
                        st.metric("Flesch-Kincaid Grade", f"Grade {s['Flesch-Kincaid Grade']:.1f}",
                                  help="Approximate U.S. school grade level")

                with viz_tab2:
                    st.markdown("#### Grade Level Comparison")
                    grade_metrics = {
                        "Flesch-Kincaid": s['Flesch-Kincaid Grade'],
                        "Gunning Fog": s['Gunning Fog'],
                        "SMOG Index": s['SMOG Index'],
                        "Coleman-Liau": s.get('Coleman-Liau Index', 0),
                        "ARI": s.get('Automated Readability Index', 0)
                    }
                    fig = go.Figure(data=[go.Bar(
                        y=list(grade_metrics.keys()),
                        x=list(grade_metrics.values()),
                        orientation='h',
                        marker_color='#636efa'
                    )])
                    fig.update_layout(height=400, xaxis_title="Grade Level", yaxis_title="Metric")
                    st.plotly_chart(fig, use_container_width=True)

                    avg_grade = sum(grade_metrics.values()) / len(grade_metrics)
                    st.metric("Average Grade Level", f"{avg_grade:.1f}")

                with viz_tab3:
                    st.markdown("#### Readability Radar Chart")
                    categories = ['Flesch Ease', 'F-K Grade', 'Gunning Fog', 'SMOG Index']
                    max_ease, max_grade = 100, 20
                    normalized = [
                        s['Flesch Reading Ease'] / max_ease * 10,
                        (max_grade - s['Flesch-Kincaid Grade']) / max_grade * 10,
                        (max_grade - s['Gunning Fog']) / max_grade * 10,
                        (max_grade - s['SMOG Index']) / max_grade * 10,
                    ]
                    categories = categories + [categories[0]]
                    normalized = normalized + [normalized[0]]

                    fig = go.Figure(go.Scatterpolar(
                        r=normalized, theta=categories, fill='toself',
                        fillcolor='rgba(99,110,250,0.3)',
                        line=dict(color='rgb(99,110,250)'),
                        name="Readability Scores"
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                                      showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Higher values indicate better readability across                    multiple dimensions.")

                with viz_tab4:
                    st.markdown("#### Text Statistics")
                    word_count = len(st.session_state["__latest_source_text"].split())
                    char_count = len(st.session_state["__latest_source_text"])
                    sentence_count = st.session_state["__latest_source_text"].count('.') + st.session_state["__latest_source_text"].count('!') + st.session_state["__latest_source_text"].count('?')
                    avg_sentence_len = word_count / max(sentence_count, 1)
                    avg_word_len = char_count / max(word_count, 1)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Words", f"{word_count}")
                        st.metric("Characters", f"{char_count}")
                    with c2:
                        st.metric("Sentences", f"{sentence_count}")
                        st.metric("Avg Sentence Length", f"{avg_sentence_len:.1f} words")
                    with c3:
                        st.metric("Avg Word Length", f"{avg_word_len:.1f} chars")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("Submit text on the left to see AI output and readability analytics here.")



# ---------- User Dashboard Page ----------
elif st.session_state.page == "dashboard":
    st.markdown(f'<h1 class="main-header" style="text-align: center;">Your Personal Dashboard</h1>', unsafe_allow_html=True)

    top_cols = st.columns([1, 4, 1])
    with top_cols[0]:
        if st.button("⬅️ Back to Home", use_container_width=True, key="back_home_top"):
            st.session_state.page = "home"
            animate_transition()
            st.rerun()
    with top_cols[2]:
        if st.button("Logout 🚪", use_container_width=True, help="Logout from your account"):
            st.session_state.user = None
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

    st.markdown("---")

    conn = sqlite3.connect("users.db")
    user_data_row = pd.read_sql_query("SELECT * FROM users WHERE username=?", conn, params=(st.session_state.user,)).iloc[0]

    tab1, tab2, tab3 = st.tabs(["👤 Profile & Security", "🤖 AI Preferences", "📊 My Analytics"])

    # --- TAB 1: Profile & Security ---
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Profile Settings</p>', unsafe_allow_html=True)

        profile_cols = st.columns([1, 2])
        with profile_cols[0]:
            st.subheader("Profile Picture")
            if user_data_row['profile_pic']:
                try:
                    image_b64 = base64.b64encode(user_data_row['profile_pic']).decode()
                    st.markdown(f'''<div style="width: 150px; height: 150px; border-radius: 50%; overflow: hidden; border: 3px solid #444;"><img src="data:image/png;base64,{image_b64}" style="width: 100%; height: 100%; object-fit: cover;"></div>''', unsafe_allow_html=True)
                except:
                    st.error("Could not load image.")
            else:
                st.info("No profile picture uploaded.")
            new_profile_pic = st.file_uploader("Update Profile Picture", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

        with profile_cols[1]:
            name = st.text_input("Name", value=user_data_row['name'] or "")
            age_category = st.selectbox("Age Category", AGE_CATEGORIES, index=AGE_CATEGORIES.index(user_data_row['age_category'] or "15 - 20"))
            lang = st.radio("Preferred Language", ["English", "Telugu", "Hindi"], index=["English", "Telugu", "Hindi"].index(user_data_row['language'] or "English"))
            # NEW: Reading Preferences and Content Type fields
            reading_prefs = st.text_input("Reading Preferences (e.g., simple, academic)", value=user_data_row['reading_preferences'] or "")
            content_type = st.text_input("Preferred Content Type (e.g., news, technical)", value=user_data_row['content_type'] or "")

        if st.button("Update Profile", use_container_width=True, type="primary"):
            cur = conn.cursor()
            profile_pic_binary = user_data_row['profile_pic']
            if new_profile_pic:
                image = Image.open(new_profile_pic)
                profile_pic_binary = image_to_binary(image)
            # NEW: Updated UPDATE statement
            cur.execute("""UPDATE users SET name=?, age_category=?, language=?, profile_pic=?,
                           reading_preferences=?, content_type=? WHERE username=?""",
                          (name, age_category, lang, profile_pic_binary,
                           reading_prefs, content_type, st.session_state.user))
            conn.commit()
            st.success("Profile updated successfully!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Change Password</p>', unsafe_allow_html=True)
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Update Password", use_container_width=True)
            if submitted:
                if hash_password(current_password) != user_data_row['password']:
                    st.error("Current password is incorrect")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("New password must be at least 6 characters long")
                else:
                    update_password(st.session_state.user, new_password)
                    st.success("Password updated successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: AI Preferences ---
    with tab2:
        # This tab's code remains unchanged
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Default AI Model</p>', unsafe_allow_html=True)
        st.info("Select the AI model you'd like to use by default on the home page.")
        active_models_df = get_active_models()
        if not active_models_df.empty:
            model_options = {f"{row['display_name']}": row['model_id'] for _, row in active_models_df.iterrows()}
            current_default_model_id = user_data_row['default_model']
            model_display_names = list(model_options.keys())
            default_index = 0
            for i, model_id in enumerate(model_options.values()):
                if model_id == current_default_model_id:
                    default_index = i
                    break
            selected_default_model_display = st.selectbox(
                "Choose your default AI Model:",
                model_display_names,
                index=default_index,
                key="default_model_selection"
            )
            if st.button("Set as Default Model", use_container_width=True, type="primary"):
                selected_model_id = model_options[selected_default_model_display]
                cur = conn.cursor()
                cur.execute("UPDATE users SET default_model=? WHERE username=?", (selected_model_id, st.session_state.user))
                conn.commit()
                st.success("Default model updated successfully!")
                st.rerun()
        else:
            st.warning("No active models are available to set as default. Please contact an administrator.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- TAB 3: My Analytics ---
    with tab3:
        # NEW: Updated query to fetch readability scores
        df = pd.read_sql_query("""
            SELECT timestamp, task_type, input_text, output_text, sentiment,
                   flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index
            FROM submissions
            WHERE username=? ORDER BY id DESC
        """, conn, params=(st.session_state.user,))

        if not df.empty:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(label="Total Submissions", value=len(df))
            with metric_cols[1]:
                most_common_sentiment = df['sentiment'].mode()[0]
                st.metric(label="Most Common Sentiment", value=most_common_sentiment)
            with metric_cols[2]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                latest_submission = df['timestamp'].max().strftime("%b %d, %Y")
                st.metric(label="Last Submission", value=latest_submission)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Submissions History</p>', unsafe_allow_html=True)

            # NEW: Reordered columns to show scores in the dataframe
            df_display = df[['timestamp', 'task_type', 'input_text', 'output_text', 'flesch_kincaid_grade', 'gunning_fog', 'smog_index']]
            # Round the scores for cleaner display
            for col in ['flesch_kincaid_grade', 'gunning_fog', 'smog_index']:
                df_display[col] = df_display[col].round(2)

            st.dataframe(df_display, use_container_width=True)

            st.markdown("---")
            export_cols = st.columns(2)
            with export_cols[0]:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Data (CSV)", data=csv, file_name="my_submissions.csv", use_container_width=True)
            with export_cols[1]:
                pdf_data = export_pdf(df_display)
                st.download_button("Download Data (PDF)", data=pdf_data, file_name="my_submissions.pdf", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("You haven't submitted any text yet. Go to the Home page to start analyzing!")

    conn.close()

# ---------- Admin Dashboard Page ----------
elif st.session_state.page == "admin":
    st.markdown('<h1 class="main-header" style="text-align: center;">Admin Dashboard</h1>', unsafe_allow_html=True)

    # --- Admin Header ---
    admin_header_cols = st.columns([3, 1, 1])
    with admin_header_cols[0]:
        st.markdown(f"#### Welcome, {st.session_state.user}!")
    with admin_header_cols[1]:
        if st.session_state.theme == "light":
            if st.button("🌙 Dark", use_container_width=True, help="Switch to dark mode"):
                st.session_state.theme = "dark"
                st.rerun()
        else:
            if st.button("☀️ Light", use_container_width=True, help="Switch to light mode"):
                st.session_state.theme = "light"
                st.rerun()
    with admin_header_cols[2]:
        if st.button("🚪 Logout", use_container_width=True, help="Logout from admin panel"):
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()

    st.markdown("---")

    conn = sqlite3.connect("users.db")

    # --- NEW: Tabbed Interface for Admin sections ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Platform Analytics", "👥 User Management", "🤖 AI Model Management", "📜 All Submissions"])

    # --- TAB 1: Platform Analytics ---
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Key Metrics</p>', unsafe_allow_html=True)
        metric_cols = st.columns(2)
        with metric_cols[0]:
            total_users = pd.read_sql_query("SELECT COUNT(*) FROM users", conn).iloc[0, 0]
            st.metric(label="Total Registered Users", value=total_users)
        with metric_cols[1]:
            total_submissions = pd.read_sql_query("SELECT COUNT(*) FROM submissions", conn).iloc[0, 0]
            st.metric(label="Total Text Submissions", value=total_submissions)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">User Demographics</p>', unsafe_allow_html=True)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            lang_counts = pd.read_sql_query("SELECT language, COUNT(*) as count FROM users GROUP BY language", conn)
            if not lang_counts.empty:
                fig = px.pie(lang_counts, values="count", names="language", title="User Language Distribution", hole=0.3)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)
        with chart_cols[1]:
            age_counts = pd.read_sql_query("SELECT age_category, COUNT(*) as count FROM users GROUP BY age_category", conn)
            if not age_counts.empty:
                fig = px.bar(age_counts, x="age_category", y="count", title="User Age Distribution")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)
        # NEW: Model Usage Pie Chart
        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI Model Usage Distribution</p>', unsafe_allow_html=True)
        model_usage_df = pd.read_sql_query("SELECT model_id_used, COUNT(*) as usage_count FROM submissions GROUP BY model_id_used", conn)
        if not model_usage_df.empty:
            fig = px.pie(model_usage_df, values="usage_count", names="model_id_used", title="Model Popularity", hole=0.3)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model usage data has been recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: User Management ---
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Manage Users</p>', unsafe_allow_html=True)
        users_df = pd.read_sql_query("SELECT username, name, email, age_category, language, is_active FROM users WHERE username NOT IN ('shashi', 'pranavi')", conn)

        if not users_df.empty:
            for _, user in users_df.iterrows():
                with st.expander(f"{user['name']} ({user['username']}) - Status: {'Active' if user['is_active'] else 'Blocked'}"):
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Age Category:** {user['age_category']}")

                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        if user['is_active']:
                            if st.button(f"Block User", key=f"block_{user['username']}", use_container_width=True):
                                toggle_user_status(user['username'], 0)
                                st.success(f"User {user['username']} has been blocked.")
                                st.rerun()
                        else:
                            if st.button(f"Unblock User", key=f"unblock_{user['username']}", use_container_width=True):
                                toggle_user_status(user['username'], 1)
                                st.success(f"User {user['username']} has been unblocked.")
                                st.rerun()
                    with btn_cols[1]:
                        if st.button(f"Delete User", key=f"delete_{user['username']}", use_container_width=True, type="primary"):
                            delete_user(user['username'])
                            st.success(f"User {user['username']} has been deleted.")
                            st.rerun()
        else:
            st.info("No users found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: AI Model Management ---
    with tab3:
        TASK_TYPES = ["summarization", "paraphrasing", "question_answering", "text_generation"]

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Add New AI Model</p>', unsafe_allow_html=True)
        with st.form("add_model_form", clear_on_submit=True):
            new_model_id = st.text_input("Model ID (from Hugging Face)", placeholder="e.g., google/pegasus-cnn_dailymail")
            new_display_name = st.text_input("Display Name", placeholder="e.g., Pegasus CNN (Summarization)")
            new_task_type = st.selectbox("Task Type", TASK_TYPES, index=0) # NEW
            new_remarks = st.text_area("Remarks/Description", placeholder="Good for news articles")
            new_status = st.selectbox("Status", ["active", "inactive"])

            if st.form_submit_button("Add Model", use_container_width=True, type="primary"):
                if new_model_id and new_display_name:
                    # Add the new model with its task type
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    cur.execute("INSERT INTO models (model_id, display_name, remarks, status, task_type) VALUES (?, ?, ?, ?, ?)",
                                (new_model_id, new_display_name, new_remarks, new_status, new_task_type))
                    conn.commit()
                    conn.close()
                    st.success("Model added successfully!")
                    st.rerun()
                else:
                    st.warning("Model ID and Display Name are required.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Manage Existing Models</p>', unsafe_allow_html=True)
        models_df = pd.read_sql_query("SELECT * FROM models", conn)
        if not models_df.empty:
            for index, model in models_df.iterrows():
                usage_count = pd.read_sql_query("SELECT COUNT(*) FROM submissions WHERE model_id_used=?", conn, params=(model['model_id'],)).iloc[0, 0]
                with st.expander(f"{model['display_name']} ({model['task_type']}) - Times Used: {usage_count}"):
                    with st.form(f"edit_model_{model['model_id']}"):
                        edited_display_name = st.text_input("Display Name", value=model['display_name'])
                        edited_task_type = st.selectbox("Task Type", TASK_TYPES, index=TASK_TYPES.index(model['task_type'])) # NEW
                        edited_remarks = st.text_area("Remarks", value=model['remarks'])
                        edited_status = st.selectbox("Status", ["active", "inactive"], index=["active", "inactive"].index(model['status']))

                        btn_cols = st.columns(2)
                        with btn_cols[0]:
                            if st.form_submit_button("Save Changes", use_container_width=True):
                                # Update the model with its task type
                                conn = sqlite3.connect("users.db")
                                cur = conn.cursor()
                                cur.execute("UPDATE models SET display_name=?, remarks=?, status=?, task_type=? WHERE model_id=?",
                                            (edited_display_name, edited_remarks, edited_status, edited_task_type, model['model_id']))
                                conn.commit()
                                conn.close()
                                st.success("Model updated successfully!")
                                st.rerun()
                        with btn_cols[1]:
                            if st.form_submit_button("Delete Model", use_container_width=True, type="primary"):
                                delete_model(model['model_id'])
                                st.rerun()
        else:
            st.info("No AI models found in the database.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 4: All Submissions ---
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">View All User Submissions</p>', unsafe_allow_html=True)
        df_sub = pd.read_sql_query("SELECT username, timestamp, task_type, model_id_used, input_text, output_text, sentiment FROM submissions ORDER BY id DESC", conn)
        st.dataframe(df_sub, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    conn.close()

# ---------- Forgot Password Page ----------
elif st.session_state.page == "forgot":
    # NEW: A modern, centered layout consistent with other pages
    st.markdown('<h1 class="main-header" style="text-align: center;">Reset Password</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Enter your details to receive a temporary password.</p>', unsafe_allow_html=True)

    # Use columns to center the form
    _, reset_col, _ = st.columns([1, 1.5, 1])

    with reset_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email", placeholder="Enter your email address")

        if st.button("Reset Password", use_container_width=True, type="primary"):
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND email=?", (username, email))
            user = cur.fetchone()

            if user:
                import random
                import string
                temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

                # Update the user's password in the database
                update_password(username, temp_password)

                st.success("Password reset successful!")
                st.info(f"Your temporary password is: **{temp_password}**")
                st.warning("Please log in and change your password immediately from the dashboard.")
            else:
                st.error("The username and email you entered do not match our records.")

            conn.close()

        # "Back to Login" button
        if st.button("⬅️ Back to Login", use_container_width=True):
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
