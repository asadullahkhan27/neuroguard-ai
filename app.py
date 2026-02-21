import streamlit as st
import requests
import os
import time
from burnout_engine import calculate_burnout_risk
from crisis_detector import detect_crisis

# -----------------------------
# Hugging Face API Setup
# -----------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

EMOTION_API = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
GEN_API = "https://api-inference.huggingface.co/models/google/flan-t5-base"


def query_model(api_url, payload, retries=3):
    for i in range(retries):
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code != 200:
            return None

        result = response.json()

        # If model is loading, wait and retry
        if isinstance(result, dict) and "error" in result:
            if "loading" in result["error"].lower():
                time.sleep(5)
                continue
            else:
                return None

        return result

    return None


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="NeuroGuard AI", layout="centered")

st.title("🧠 NeuroGuard AI")
st.caption("Early Mental Burnout & Depression Detection System")

st.info(
    "This tool is not a medical diagnosis system. "
    "If you are experiencing severe distress, please consult a licensed professional."
)

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

user_input = st.text_area("Describe how you are feeling today:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    # -----------------------------
    # Emotion Detection
    # -----------------------------

    with st.spinner("Analyzing emotion..."):

        emotion_result = query_model(EMOTION_API, {"inputs": user_input})

        if emotion_result is None:
            st.error("Emotion model unavailable. Please try again in a few seconds.")
            st.stop()

        # HF emotion model sometimes returns list inside list
        if isinstance(emotion_result, list):
            emotion = emotion_result[0]["label"]
            confidence = emotion_result[0]["score"]
        else:
            st.error("Unexpected response from emotion model.")
            st.stop()

    st.session_state.emotion_history.append(emotion)

    # -----------------------------
    # Crisis Detection
    # -----------------------------

    crisis_flag = detect_crisis(user_input)

    # -----------------------------
    # Burnout Risk
    # -----------------------------

    risk = calculate_burnout_risk(
        emotion,
        confidence,
        st.session_state.emotion_history
    )

    # -----------------------------
    # Generative Response
    # -----------------------------

    with st.spinner("Generating supportive response..."):

        prompt = f"""
        You are a compassionate AI mental health assistant.
        User emotion: {emotion}
        User message: {user_input}

        Generate an empathetic support message with gentle coping advice.
        """

        gen_result = query_model(GEN_API, {"inputs": prompt})

        if gen_result is None:
            ai_response = "Support response temporarily unavailable."
        elif isinstance(gen_result, list):
            ai_response = gen_result[0]["generated_text"]
        else:
            ai_response = "Support response temporarily unavailable."

    # -----------------------------
    # Display Results
    # -----------------------------

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
        st.error(
            "Crisis-related language detected. "
            "Please consider seeking professional help immediately."
        )

    st.subheader("AI Support Response")
    st.write(ai_response)

    # -----------------------------
    # Emotion Trend Chart
    # -----------------------------

    st.subheader("Emotion Trend")

    emotion_counts = {}
    for e in st.session_state.emotion_history:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    st.bar_chart(emotion_counts)
