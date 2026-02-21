crisis_keywords = [
    "suicide",
    "kill myself",
    "hopeless",
    "ending my life",
    "worthless",
    "no reason to live"
]

def detect_crisis(text):
    for word in crisis_keywords:
        if word in text.lower():
            return True
    return False