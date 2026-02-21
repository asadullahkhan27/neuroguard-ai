import streamlit as st
from transformers import pipeline

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="NeuroGuard AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroGuard AI")
st.subheader("Early Mental Burnout & Emotion Detection System")

st.info(
    "This tool is not a medical diagnosis system. "
    "If you are experiencing severe distress, please consult a licensed professional."
)

# ----------------------------
# Load Sentiment Model (Local)
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# ----------------------------
# Emotion Analysis Function
# ----------------------------
def analyze_emotion(text):
    try:
        result = sentiment_model(text)[0]
        return result["label"], result["score"]
    except:
        return None, None

# ----------------------------
# Support Response Generator
# ----------------------------
def generate_response(label):
    responses = {
        "POSITIVE": "It’s good to see some positivity. Keep building on the things that are helping you feel this way.",
        "NEGATIVE": "It seems like you're going through something difficult. Try taking a short break, talking to someone you trust, or doing a small activity that helps you relax."
    }
    return responses.get(label, "Thank you for sharing how you feel.")

# ----------------------------
# UI
# ----------------------------
user_input = st.text_area(
    "Describe how you are feeling today:",
    height=120
)

if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter something before analyzing.")
    else:
        with st.spinner("Analyzing emotion..."):
            label, score = analyze_emotion(user_input)

        if label is None:
            st.error("Analysis failed. Please try again.")
        else:

            confidence = round(score * 100, 2)

            if label == "POSITIVE":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#d4edda;
                        padding:20px;
                        border-radius:10px;
                        border:1px solid #28a745;">
                        <h3 style="color:#155724;">Detected Sentiment: {label}</h3>
                        <p><strong>Confidence:</strong> {confidence}%</p>
                        <p>It’s good to see positivity. Keep building on what’s helping you feel this way.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif label == "NEGATIVE":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f8d7da;
                        padding:20px;
                        border-radius:10px;
                        border:1px solid #dc3545;">
                        <h3 style="color:#721c24;">Detected Sentiment: {label}</h3>
                        <p><strong>Confidence:</strong> {confidence}%</p>
                        <p>You may be experiencing emotional strain. Consider taking a short break, talking to someone you trust, or doing something calming.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.info("Thank you for sharing how you feel.")
