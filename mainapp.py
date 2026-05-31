import streamlit as st
import joblib
from text_utils import clean_text
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Patient Condition Classifier",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Patient Condition Classification")
st.write("Enter a drug review to predict the medical condition.")

# -----------------------------
# Load Model Package
# -----------------------------
@st.cache_resource
def load_models():
    package = joblib.load("drug_condition_models.pkl")
    return package

package = load_models()
tfidf = package["tfidf"]
label_encoder = package["label_encoder"]
models = package["models"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Models trained and Supported Conditions")
st.sidebar.write("• Depression")
st.sidebar.write("• High Blood Pressure")
st.sidebar.write("• Type 2 Diabetes")



st.sidebar.header("Examples")
example_review = st.sidebar.selectbox(
    "Select an example",
    [
        "",
        "I was feeling very low and lost interest in everything. This medicine improved my mood.",
        "My blood pressure was high but after taking this medication daily, my readings are now normal.",
        "My blood sugar levels were very high before treatment. After taking this medication regularly, my glucose levels are well controlled.",
	"I had constant sadness, poor sleep, and negative thoughts. This treatment helped stabilize my mood and reduced my depressive symptoms.",
	"I had high blood pressure for years. Since starting this medicine, my BP levels have become stable and I feel much better.",
	"I have type 2 diabetes and this drug helped reduce my blood sugar levels."
    ]
)


# -----------------------------
# Model Selection
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[model_name]

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter Review Here:",
    value=example_review,
    height=150
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Condition"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Clean text
        cleaned = clean_text(user_input)

        # Vectorize
        vectorized = tfidf.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)
        predicted_condition = label_encoder.inverse_transform(prediction)[0]

        # Display prediction
        st.success(f"Predicted Condition: **{predicted_condition}**")

        # -----------------------------
        # Confidence Score
        # -----------------------------
        try:
            # For models with probability support
            probs = model.predict_proba(vectorized)
            confidence = np.max(probs) * 100
            st.info(f"Percentage: {confidence:.2f}%")

        except:
            # For LinearSVM (no predict_proba)
            try:
                score = model.decision_function(vectorized)
                confidence = np.max(score)
                st.info(f"Model Score: {confidence:.2f}")
            except:
                st.info("Confidence score not available for this model.")