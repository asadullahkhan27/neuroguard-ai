import streamlit as st
import requests
import os
from burnout_engine import calculate_burnout_risk
from crisis_detector import detect_crisis
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table
from reportlab.lib.units import inch

# -------------------------
# Hugging Face API Setup
# -------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

EMOTION_API = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
GEN_API = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def query_model(api_url, payload):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="NeuroGuard AI", layout="centered")

st.title("🧠 NeuroGuard AI")
st.caption("Early Mental Burnout & Depression Detection System")

st.info("This tool is not a medical diagnosis system. If you are in severe distress, please contact a licensed professional.")

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

user_input = st.text_area("Describe how you are feeling today:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:

        # -------------------------
        # Emotion Detection
        # -------------------------
        emotion_result = query_model(EMOTION_API, {"inputs": user_input})

        if isinstance(emotion_result, dict) and "error" in emotion_result:
            st.error("Model loading. Please wait 10 seconds and try again.")
            st.stop()

        emotion = emotion_result[0]["label"]
        confidence = emotion_result[0]["score"]

        st.session_state.emotion_history.append(emotion)

        # -------------------------
        # Crisis Detection
        # -------------------------
        crisis_flag = detect_crisis(user_input)

        # -------------------------
        # Burnout Risk
        # -------------------------
        risk = calculate_burnout_risk(
            emotion,
            confidence,
            st.session_state.emotion_history
        )

        # -------------------------
        # Generative AI Response
        # -------------------------
        prompt = f"""
        You are a compassionate AI mental health assistant.
        User emotion: {emotion}
        User message: {user_input}

        Generate an empathetic support message with gentle coping advice.
        """

        gen_result = query_model(GEN_API, {"inputs": prompt})

        if isinstance(gen_result, list):
            ai_response = gen_result[0]["generated_text"]
        else:
            ai_response = "Support response temporarily unavailable."

        # -------------------------
        # Display Results
        # -------------------------

        st.subheader("Emotion Analysis")
        st.write(f"Detected Emotion: **{emotion}**")
        st.progress(float(confidence))

        st.subheader("Burnout Risk Level")

        if risk == "High Risk":
            st.error(risk)
        elif risk == "Moderate Risk":
            st.warning(risk)
        else:
            st.success(risk)

        if crisis_flag:
            st.error("Crisis-related language detected. Please seek professional help immediately.")

        st.subheader("AI Support Response")
        st.write(ai_response)

        # -------------------------
        # Emotion Trend Dashboard
        # -------------------------

        st.subheader("Emotion Trend")

        emotion_counts = {}
        for e in st.session_state.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        st.bar_chart(emotion_counts)

        # -------------------------
        # Generate PDF Report
        # -------------------------

        if st.button("Download Mental Health Report"):

            file_path = "NeuroGuard_Report.pdf"
            doc = SimpleDocTemplate(file_path)
            elements = []

            styles = getSampleStyleSheet()
            elements.append(Paragraph("NeuroGuard AI Mental Health Report", styles['Heading1']))
            elements.append(Spacer(1, 0.5 * inch))

            data = [
                ["Emotion", emotion],
                ["Confidence", f"{confidence:.2f}"],
                ["Burnout Risk", risk]
            ]

            table = Table(data)
            table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])

            elements.append(table)
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph("AI Support Response:", styles['Heading2']))
            elements.append(Paragraph(ai_response, styles['Normal']))

            doc.build(elements)

            with open(file_path, "rb") as f:
                st.download_button(
                    label="Click to Download Report",
                    data=f,
                    file_name="NeuroGuard_Report.pdf",
                    mime="application/pdf"
                )