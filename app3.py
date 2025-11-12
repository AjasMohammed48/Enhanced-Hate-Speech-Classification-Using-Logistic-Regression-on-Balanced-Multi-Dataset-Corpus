import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("toxic_logreg_model1.pkl")
vectorizer = joblib.load("toxic_vectorizer1.pkl")

# Text cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

# Streamlit setup
st.set_page_config(page_title="Toxic Speech Detector", page_icon="🧠")
st.title("🧠 Toxic Speech Detection (Human-Style Response)")
st.write("AI system that classifies text as **Hate Speech**, **Offensive**, or **Neutral**, and explains why.")

# User input
user_input = st.text_area("💬 Enter a tweet or message:")

if st.button("Analyze"):
    if user_input.strip():
        # Clean + vectorize
        cleaned = clean_text(user_input)
        X_input = vectorizer.transform([cleaned])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # Labels and messages
        labels = ["Hate Speech", "Offensive", "Neutral"]
        predicted_label = labels[int(prediction)]

        # 🧠 Natural language explanations
        explanations = {
            "Hate Speech": "This sentence expresses **hateful or discriminatory language** targeting a group or identity.",
            "Offensive": "This sentence contains **offensive, insulting, or disrespectful language.**",
            "Neutral": "This sentence appears **neutral and respectful**, with no signs of toxicity."
        }

        # 🎯 Output
        st.markdown(f"### 🧩 Classification: **{predicted_label}**")
        st.markdown(f"🗣️ {explanations[predicted_label]}")

        # Confidence display
        st.markdown("#### 🔍 Confidence Scores:")
        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))

        # Color-coded message
        if predicted_label == "Hate Speech":
            st.error("🚨 High likelihood of **hate or racism** detected.")
        elif predicted_label == "Offensive":
            st.warning("⚠️ Contains **rude or inappropriate tone.**")
        else:
            st.success("✅ Looks clean and **non-toxic.**")

    else:
        st.warning("Please enter some text to analyze.")
