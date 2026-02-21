import streamlit as st
import requests
import os

# -----------------------------
# Page Config
# -----------------------------
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

# -----------------------------
# Hugging Face Setup
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# -----------------------------
# Emotion Analysis Function
# -----------------------------
def analyze_emotion(text):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                emotions = result[0]
                top_emotion = max(emotions, key=lambda x: x["score"])
                return top_emotion["label"], top_emotion["score"]
            else:
                return None, None

        elif response.status_code == 503:
            return "Model warming up", None

        else:
            return None, None

    except Exception:
        return None, None


# -----------------------------
# Burnout Response Generator
# -----------------------------
def generate_response(emotion):
    responses = {
        "sadness": "It sounds like you're feeling low. Try taking a short break, talking to someone you trust, or doing something small that usually helps you feel better.",
        "anger": "You may be feeling frustrated. Deep breathing or stepping away from the situation for a few minutes can help reset your emotions.",
        "fear": "It seems like anxiety or worry might be present. Try grounding techniques like focusing on your breathing or listing 5 things you can see.",
        "joy": "That’s great to hear. Keep doing what’s bringing you positive energy.",
        "neutral": "Thanks for sharing how you're feeling. Being aware of your emotions is an important first step."
    }

    return responses.get(
        emotion.lower(),
        "Thanks for sharing. Consider taking some time for self-care today."
    )


# -----------------------------
# UI Input
# -----------------------------
user_input = st.text_area(
    "Describe how you are feeling today:",
    height=120
)

if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter something before analyzing.")
    else:
        with st.spinner("Analyzing emotion..."):
            emotion, score = analyze_emotion(user_input)

        if emotion is None:
            st.error("Emotion service temporarily unavailable. Please try again later.")

        elif emotion == "Model warming up":
            st.warning("Model is warming up. Please try again in a few seconds.")

        else:
            st.success(f"Detected Emotion: {emotion.capitalize()}")

            response_text = generate_response(emotion)

            st.markdown("### Suggested Support")
            st.write(response_text)
