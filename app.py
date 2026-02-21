import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="NeuroGuard AI Pro", layout="wide")

# ----------------------------------
# CUSTOM UI STYLING
# ----------------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #ffffff;}
.stButton>button {
    background-color:#1f77b4;
    color:white;
    border-radius:8px;
}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.title("NeuroGuard AI – Pro Max")
st.caption("Advanced Mental Burnout & Depression Intelligence System")

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Emotion Analysis", "AI Chatbot", "Reports", "Therapist Finder", "Emergency Support"]
)

# ----------------------------------
# LOAD MODELS
# ----------------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    generator = pipeline("text-generation", model="distilgpt2")
    return sentiment, generator

sentiment_model, chatbot_model = load_models()

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------
# FUNCTIONS
# ----------------------------------
def analyze(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

def calculate_scores(label, confidence):
    burnout = int(confidence * 100) if label == "NEGATIVE" else 15
    depression = int(confidence * 100) if label == "NEGATIVE" else 10
    wellness = 100 - burnout
    return burnout, depression, wellness

# ----------------------------------
# DASHBOARD
# ----------------------------------
if menu == "Dashboard":

    st.header("Mental Health Overview")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Burnout", int(df["burnout"].mean()))
        col2.metric("Avg Depression Risk", int(df["depression"].mean()))
        col3.metric("Avg Wellness Score", int(df["wellness"].mean()))

        st.subheader("Burnout Trend")
        fig, ax = plt.subplots()
        ax.plot(df["burnout"])
        ax.set_ylabel("Burnout Score")
        st.pyplot(fig)
    else:
        st.info("No data yet. Go to Emotion Analysis.")

# ----------------------------------
# EMOTION ANALYSIS
# ----------------------------------
if menu == "Emotion Analysis":

    st.header("Emotion & Burnout Analysis")

    user_input = st.text_area("Describe how you feel today")

    if st.button("Analyze Now"):

        if user_input.strip():

            label, confidence = analyze(user_input)
            burnout, depression, wellness = calculate_scores(label, confidence)

            st.session_state.history.append({
                "date": datetime.date.today(),
                "label": label,
                "burnout": burnout,
                "depression": depression,
                "wellness": wellness
            })

            col1, col2 = st.columns(2)

            if label == "POSITIVE":
                col1.success(f"Positive Mood ({round(confidence*100,2)}%)")
            else:
                col1.error(f"Negative Mood ({round(confidence*100,2)}%)")

            col2.metric("Wellness Score", wellness)

            st.subheader("Burnout Risk Level")
            st.progress(burnout/100)

            if burnout > 70:
                st.error("High Burnout Risk")
            elif burnout > 40:
                st.warning("Moderate Burnout Risk")
            else:
                st.success("Low Burnout Risk")

            st.subheader("Depression Probability")
            st.metric("Risk %", depression)

# ----------------------------------
# AI CHATBOT
# ----------------------------------
if menu == "AI Chatbot":

    st.header("AI Mental Support Assistant")

    chat_input = st.text_input("Talk to NeuroGuard AI")

    if st.button("Generate Response"):
        if chat_input:
            response = chatbot_model(chat_input, max_length=120)[0]["generated_text"]
            st.write(response)

# ----------------------------------
# REPORTS
# ----------------------------------
if menu == "Reports":

    st.header("Generate Mental Health Report")

    if st.button("Create PDF Report"):

        if not st.session_state.history:
            st.warning("No data available.")
        else:
            doc = SimpleDocTemplate("NeuroGuard_Report.pdf", pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("NeuroGuard AI Report", styles["Heading1"]))
            elements.append(Spacer(1, 0.5 * inch))

            for entry in st.session_state.history:
                text = f"""
                Date: {entry['date']}<br/>
                Mood: {entry['label']}<br/>
                Burnout: {entry['burnout']}<br/>
                Depression Risk: {entry['depression']}%<br/>
                Wellness Score: {entry['wellness']}
                """
                elements.append(Paragraph(text, styles["Normal"]))
                elements.append(Spacer(1, 0.3 * inch))

            doc.build(elements)

            with open("NeuroGuard_Report.pdf", "rb") as file:
                st.download_button("Download Report", file, "NeuroGuard_Report.pdf")

# ----------------------------------
# THERAPIST FINDER
# ----------------------------------
if menu == "Therapist Finder":

    st.header("Find Professional Support")

    city = st.selectbox("Select City", ["Lahore", "Karachi", "Islamabad"])

    if city == "Lahore":
        st.info("Lahore Psychology Center\nContact: 042-XXXXXXX")
    elif city == "Karachi":
        st.info("Karachi Mental Wellness Clinic\nContact: 021-XXXXXXX")
    else:
        st.info("Islamabad Therapy Services\nContact: 051-XXXXXXX")

# ----------------------------------
# EMERGENCY SUPPORT
# ----------------------------------
if menu == "Emergency Support":

    st.header("Immediate Help Resources")

    st.error("If you are experiencing severe distress, please seek immediate help.")

    st.write("""
    Pakistan Helpline: 1166  
    Edhi Ambulance: 115  
    Contact a trusted person immediately.
    """)

# ----------------------------------
# RESET DATA
# ----------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Data"):
    st.session_state.history = []
    st.sidebar.success("Session Reset Complete")
