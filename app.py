import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import speech_recognition as sr
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import os
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="NeuroGuard AI", layout="centered")
st.title("🧠 NeuroGuard AI")
st.subheader("Advanced Mental Burnout & Depression Detection System")

st.info("This system provides AI-based insights and is not a medical diagnosis tool.")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    generator = pipeline("text-generation", model="distilgpt2")
    return sentiment, generator

sentiment_model, chatbot_model = load_models()

# -----------------------------
# SESSION STORAGE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

def depression_probability(label, confidence):
    if label == "NEGATIVE":
        return round(confidence * 100, 2)
    else:
        return round((1 - confidence) * 40, 2)

# -----------------------------
# VOICE INPUT
# -----------------------------
st.markdown("## 🎤 Voice Emotion Detection")

audio_file = st.file_uploader("Upload voice file (wav format)", type=["wav"])

if audio_file:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success("Transcribed Text:")
            st.write(text)
        except:
            st.error("Could not transcribe audio")

# -----------------------------
# TEXT INPUT
# -----------------------------
st.markdown("## 📝 Text Emotion Analysis")

user_input = st.text_area("Describe how you feel today:")

if st.button("Analyze Emotion"):

    if user_input.strip():
        label, confidence = analyze(user_input)
        burnout = int(confidence * 100) if label == "NEGATIVE" else 20
        mental_score = 100 - burnout
        depression_prob = depression_probability(label, confidence)

        st.session_state.history.append({
            "date": datetime.date.today(),
            "label": label,
            "burnout": burnout,
            "mental_score": mental_score,
            "depression_prob": depression_prob
        })

        # Color Box
        if label == "POSITIVE":
            st.success(f"Positive Mood Detected ({round(confidence*100,2)}%)")
        else:
            st.error(f"Negative Mood Detected ({round(confidence*100,2)}%)")

        # Burnout
        st.markdown("### 🔥 Burnout Risk")
        st.progress(burnout/100)

        # Depression Probability
        st.markdown("### 📉 Depression Probability")
        st.metric("Risk %", depression_prob)

# -----------------------------
# CHATBOT MODE
# -----------------------------
st.markdown("## 🤖 AI Support Chatbot")

chat_input = st.text_input("Talk to NeuroGuard AI")

if st.button("Send to Chatbot"):
    if chat_input:
        response = chatbot_model(chat_input, max_length=100)[0]["generated_text"]
        st.write(response)

# -----------------------------
# PDF REPORT GENERATION
# -----------------------------
st.markdown("## 📄 Download Mental Health Report")

if st.button("Generate PDF Report"):

    doc = SimpleDocTemplate("mental_report.pdf", pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("NeuroGuard AI Mental Health Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.5 * inch))

    for entry in st.session_state.history:
        text = f"""
        Date: {entry['date']}<br/>
        Mood: {entry['label']}<br/>
        Burnout Score: {entry['burnout']}<br/>
        Mental Score: {entry['mental_score']}<br/>
        Depression Risk: {entry['depression_prob']}%
        """
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)

    with open("mental_report.pdf", "rb") as file:
        st.download_button("Download Report", file, "NeuroGuard_Report.pdf")

# -----------------------------
# EMAIL ALERT SYSTEM
# -----------------------------
st.markdown("## 📧 Email Alert System")

email_input = st.text_input("Enter Email for Alerts")

if st.button("Send Alert"):
    if email_input:
        try:
            msg = MIMEText("NeuroGuard Alert: High emotional distress detected.")
            msg["Subject"] = "NeuroGuard AI Alert"
            msg["From"] = "your_email@gmail.com"
            msg["To"] = email_input

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login("your_email@gmail.com", "your_app_password")
            server.sendmail("your_email@gmail.com", email_input, msg.as_string())
            server.quit()

            st.success("Alert Email Sent")
        except:
            st.error("Email configuration required.")

# -----------------------------
# THERAPIST RECOMMENDATION
# -----------------------------
st.markdown("## 🏥 Therapist Recommendation")

city = st.selectbox("Select Your City", ["Lahore", "Karachi", "Islamabad"])

if city:
    if city == "Lahore":
        st.write("Suggested: Lahore Psychology Center")
    elif city == "Karachi":
        st.write("Suggested: Karachi Mental Health Clinic")
    else:
        st.write("Suggested: Islamabad Therapy Services")

# -----------------------------
# HISTORY GRAPH
# -----------------------------
if st.session_state.history:
    st.markdown("## 📊 Mood Trend")

    df = pd.DataFrame(st.session_state.history)
    fig, ax = plt.subplots()
    ax.plot(df["burnout"])
    ax.set_ylabel("Burnout Score")
    st.pyplot(fig)
