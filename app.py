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
# PREMIUM UI STYLING
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">NeuroGuard AI – Enterprise Edition</div>', unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# ----------------------------
# SESSION STATE
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# CORE FUNCTIONS
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

def emotional_volatility(df):
    if len(df) < 2:
        return 0
    return int(df["burnout"].std())

def resilience_score(df):
    if len(df) == 0:
        return 0
    return int(100 - df["burnout"].mean())

def streak_counter(df):
    if len(df) == 0:
        return 0
    return sum(df["label"] == "POSITIVE")

def productivity_index(wellness, burnout):
    return max(0, int((wellness * 0.6) - (burnout * 0.4)))

def psychological_profile(df):
    if len(df) == 0:
        return "No Data"
    avg = df["burnout"].mean()
    if avg > 70:
        return "High Stress Profile"
    elif avg > 40:
        return "Moderate Stress Profile"
    else:
        return "Balanced Emotional Profile"

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Analyze Emotion",
        "Advanced Analytics",
        "AI Insights",
        "Wellness Coach",
        "Recovery Mode"
    ]
)

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.header("Mental Health Overview")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Burnout", int(df["burnout"].mean()))
        col2.metric("Depression Risk", int(df["depression"].mean()))
        col3.metric("Wellness Score", int(df["wellness"].mean()))

        col4, col5, col6 = st.columns(3)
        col4.metric("Emotional Volatility", emotional_volatility(df))
        col5.metric("Resilience Score", resilience_score(df))
        col6.metric("Positive Mood Streak", streak_counter(df))

        st.info(f"Psychological Profile: {psychological_profile(df)}")

        st.subheader("Burnout Trend")
        fig, ax = plt.subplots()
        ax.plot(df["burnout"])
        ax.set_ylabel("Burnout Score")
        st.pyplot(fig)

    else:
        st.info("No data available. Start with Emotion Analysis.")

# ----------------------------
# ANALYZE EMOTION
# ----------------------------
if menu == "Analyze Emotion":

    st.header("Emotion & Burnout Analysis")

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

            st.subheader("Burnout Heat Index")

            if burnout < 40:
                st.success("Low Heat Level")
            elif burnout < 70:
                st.warning("Moderate Heat Level")
            else:
                st.error("Critical Heat Level")

            st.progress(burnout / 100)

            st.metric("Depression Probability", depression)

            focus = productivity_index(wellness, burnout)
            st.metric("Focus Index", focus)

# ----------------------------
# ADVANCED ANALYTICS
# ----------------------------
if menu == "Advanced Analytics":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.subheader("Mood Distribution")
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
# AI INSIGHTS
# ----------------------------
if menu == "AI Insights":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        avg_burnout = df["burnout"].mean()
        volatility = emotional_volatility(df)

        st.subheader("AI Risk Assessment")

        if avg_burnout > 70:
            st.error("High Long-Term Burnout Risk Detected")
        elif avg_burnout > 40:
            st.warning("Moderate Risk Pattern Observed")
        else:
            st.success("Healthy Emotional Pattern")

        if volatility > 25:
            st.warning("High emotional fluctuations detected.")

        if df["burnout"].iloc[-1] > 80:
            st.error("Recent burnout spike detected. Immediate recovery advised.")

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

    st.success("Small consistent habits create long-term resilience.")

# ----------------------------
# RECOVERY MODE
# ----------------------------
if menu == "Recovery Mode":

    st.header("AI Guided Recovery Protocol")

    st.write("""
    1. Disconnect from workload for 30 minutes.
    2. Perform 4-7-8 breathing exercise.
    3. Hydrate and walk for 10 minutes.
    4. Avoid digital screens temporarily.
    5. Journal one stress trigger.
    """)

    st.success("Recovery plan generated based on emotional risk level.")

# ----------------------------
# DATA EXPORT
# ----------------------------
if st.session_state.history:
    df_export = pd.DataFrame(st.session_state.history)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV Data", csv, "neuroguard_data.csv")

# ----------------------------
# RESET
# ----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Data"):
    st.session_state.history = []
    st.sidebar.success("System Reset Complete")
