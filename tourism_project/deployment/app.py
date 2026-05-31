import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

# Retrieve serialized execution pipeline from your Model Registry
@st.cache_resource
def load_prediction_pipeline():
    model_path = hf_hub_download(
        repo_id="divyarathod112/tourism-package-prediction-model", # <-- Replace with your HF username
        filename="best_tourism_package_prediction_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_prediction_pipeline()

st.title("Tourism Package Prediction")
st.write("Target potential buyers efficiently. Enter customer profiles below to predict conversion likelihood before calling them.")

st.subheader("Customer Profile Context")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], help="1: Highest Tier, 3: Lowest Tier")
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    num_person = st.number_input("Number of Persons Visiting", 1, 10, 2)
    prop_star = st.slider("Preferred Property Star Rating", 3, 5, 3)
    num_trips = st.number_input("Annual Number of Trips Taken", 0, 20, 2)
    passport = st.selectbox("Has Valid Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    children = st.number_input("Number of Children Visiting (<5 yrs)", 0, 5, 0)

st.subheader("Interaction & Marketing Data")
col3, col4 = st.columns(2)

with col3:
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (Gross)", min_value=0, value=25000)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

with col4:
    prod_pitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic", "Super Deluxe", "King"])
    followups = st.slider("Number of Follow-ups Conducted", 1, 10, 3)
    duration = st.number_input("Duration of Pitch (Minutes)", min_value=0, value=15)

# Map human-readable input forms directly back to model arrays
input_df = pd.DataFrame([{
    'Age': age, 'TypeofContact': typeof_contact, 'CityTier': city_tier, 'Occupation': occupation,
    'Gender': gender, 'NumberOfPersonVisiting': num_person, 'PreferredPropertyStar': prop_star,
    'MaritalStatus': marital_status, 'NumberOfTrips': num_trips, 'Passport': 1 if passport == "Yes" else 0,
    'OwnCar': 1 if own_car == "Yes" else 0, 'NumberOfChildrenVisiting': children, 'Designation': designation,
    'MonthlyIncome': monthly_income, 'PitchSatisfactionScore': pitch_score, 'ProductPitched': prod_pitched,
    'NumberOfFollowups': followups, 'DurationOfPitch': duration
}])

st.markdown("---")
if st.button("Analyze Customer Conversion Potential", type="primary"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"**High Conversion Potential!** Probability of purchasing: {probabilities[1]:.2%}")
        st.balloons()
    else:
        st.warning(f"**Low Conversion Potential.** Probability of purchasing: {probabilities[1]:.2%}")
