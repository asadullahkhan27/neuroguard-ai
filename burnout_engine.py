negative_emotions = ["sadness", "anger", "fear"]

def calculate_burnout_risk(emotion, confidence, history):
    score = 0

    # Emotion weight
    if emotion in negative_emotions:
        score += confidence * 50

    # Repeated negative pattern
    negative_count = sum(1 for e in history if e in negative_emotions)
    score += negative_count * 10

    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Moderate Risk"
    else:
        return "Low Risk"