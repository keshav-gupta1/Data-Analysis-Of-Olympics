import re
import joblib
import numpy as np

# Load model and encoders
data = joblib.load("advanced_model.pkl")
model = data["model"]
le_sport = data["le_sport"]
le_medal = data["le_medal"]

# Simple list of sports (for matching from text)
SPORTS = list(le_sport.classes_)

def ai_chat_response(user_query):
    """AI logic to extract details and predict"""
    try:
        # Extract numbers
        age_match = re.search(r'(\d+)\s*(?:years?|yrs?)', user_query)
        height_match = re.search(r'(\d+)\s*(?:cm|centimeter)', user_query)
        weight_match = re.search(r'(\d+)\s*(?:kg|kilogram)', user_query)

        age = int(age_match.group(1)) if age_match else 25
        height = int(height_match.group(1)) if height_match else 175
        weight = int(weight_match.group(1)) if weight_match else 70

        # Detect sport from text
        sport = "unknown"
        for s in SPORTS:
            if s.lower() in user_query.lower():
                sport = s
                break

        if sport == "unknown":
            return "❌ I couldn’t detect the sport. Please mention it (e.g., swimming, football)."

        sport_code = le_sport.transform([sport])[0]
        sex = 1  # assume male
        input_data = np.array([[age, height, weight, sex, sport_code]])

        # Predict
        prediction = model.predict(input_data)
        medal = le_medal.inverse_transform(prediction)[0]

        return f"🏅 Based on your profile, you have a high chance of winning **{medal}** in {sport.capitalize()}!"
    except Exception as e:
        return f"⚠ Error: {str(e)}"
