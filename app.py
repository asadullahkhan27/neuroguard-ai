import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="NeuroGuard AI Enterprise", layout="wide")

# ----------------------------
# ADVANCED UI STYLING
# ----------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    margin-bottom:20px;
}
.metric-label {
    font-size:16px;
    color:#9ca3af;
}
.metric-value {
    font-size:28px;
    font-weight:bold;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">NeuroGuard AI – Enterprise Edition</div>', unsafe_allow_html=True)

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# ANALYSIS LOGIC
# ----------------------------
def analyze(text):
    result = model(text)[0]
    return result["label"], result["score"]

def compute_scores(label, confidence):
    burnout = int(confidence * 100) if label == "NEGATIVE" else 20
    depression = int(confidence * 100) if label == "NEGATIVE" else 15
    wellness = 100 - burnout
    stability = max(0, 100 - abs(50 - wellness))
    return burnout, depression, wellness, stability

# ----------------------------
# SIDEBAR
# ----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Analyze Emotion", "Advanced Analytics", "AI Insights", "Wellness Coach"]
)

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.header("Mental Health Overview")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        avg_burnout = int(df["burnout"].mean())
        avg_depression = int(df["depression"].mean())
        avg_wellness = int(df["wellness"].mean())

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Burnout", avg_burnout)
        col2.metric("Depression Risk", avg_depression)
        col3.metric("Wellness Score", avg_wellness)

        st.subheader("Burnout Trend")
        fig, ax = plt.subplots()
        ax.plot(df["burnout"])
        ax.set_ylabel("Burnout")
        st.pyplot(fig)

    else:
        st.info("Start analyzing emotions to see dashboard insights.")

# ----------------------------
# ANALYZE EMOTION
# ----------------------------
if menu == "Analyze Emotion":

    text = st.text_area("How are you feeling today?")

    if st.button("Run AI Analysis"):

        if text.strip():

            label, confidence = analyze(text)
            burnout, depression, wellness, stability = compute_scores(label, confidence)

            st.session_state.history.append({
                "date": datetime.date.today(),
                "label": label,
                "burnout": burnout,
                "depression": depression,
                "wellness": wellness,
                "stability": stability
            })

            col1, col2 = st.columns(2)

            if label == "POSITIVE":
                col1.success(f"Positive Mood ({round(confidence*100,2)}%)")
            else:
                col1.error(f"Negative Mood ({round(confidence*100,2)}%)")

            col2.metric("Wellness Score", wellness)

            st.progress(burnout / 100)

            if burnout > 75:
                st.error("Critical Burnout Level")
            elif burnout > 50:
                st.warning("Moderate Burnout Level")
            else:
                st.success("Stable Emotional State")

# ----------------------------
# ADVANCED ANALYTICS
# ----------------------------
if menu == "Advanced Analytics":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.subheader("Emotional Distribution")

        mood_counts = df["label"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(mood_counts, labels=mood_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)

        st.subheader("Stability Index Trend")

        fig2, ax2 = plt.subplots()
        ax2.plot(df["stability"])
        ax2.set_ylabel("Stability Index")
        st.pyplot(fig2)

    else:
        st.info("No analytics data yet.")

# ----------------------------
# AI INSIGHTS ENGINE
# ----------------------------
if menu == "AI Insights":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)
        avg_burnout = df["burnout"].mean()

        st.subheader("AI Risk Assessment")

        if avg_burnout > 70:
            st.error("High Long-Term Burnout Risk Detected")
        elif avg_burnout > 40:
            st.warning("Moderate Risk Pattern Observed")
        else:
            st.success("Healthy Emotional Pattern")

        st.subheader("System Insight")

        st.write(
            "Based on your emotional patterns, the system recommends maintaining balanced work schedules, prioritizing rest, and monitoring stress triggers."
        )

    else:
        st.info("Run emotion analysis first.")

# ----------------------------
# WELLNESS COACH
# ----------------------------
if menu == "Wellness Coach":

    st.subheader("Personalized Wellness Guidance")

    st.write("""
    • Maintain consistent sleep schedule  
    • Take micro-breaks during work  
    • Practice breathing exercises  
    • Stay socially connected  
    • Engage in light physical activity  
    """)

    st.success("Small consistent habits create long-term mental resilience.")

# ----------------------------
# RESET
# ----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Data"):
    st.session_state.history = []
    st.sidebar.success("System Reset Complete")
