
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained tourism prediction model

model_path = hf_hub_download(repo_id="Narasi/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the **Wellness Tourism Package**
based on customer characteristics such as age, city tier, income, travel behavior, and interaction data.
Please enter the customer details below to get a prediction.
""")

# Customer Details Section
st.header("Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Student", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Manager", "Senior Manager", "Director", "VP", "Entry Level", "Other"])

monthly_income = st.number_input("Monthly Income (USD)", min_value=0, max_value=500000, value=50000, step=1000)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_children_visiting = st.number_input("Number of Children (<5) Visiting", min_value=0, max_value=5, value=0, step=1)
number_of_trips = st.number_input("Average Number of Trips Per Year", min_value=0, max_value=20, value=2, step=1)

passport = st.selectbox("Has Valid Passport", ["No", "Yes"])
own_car = st.selectbox("Owns Car", ["No", "Yes"])
preferred_property_star = st.number_input("Preferred Property Star Rating", min_value=1, max_value=5, value=3, step=1)

# Customer Interaction Data Section
st.header("Customer Interaction Data")

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Premium", "Luxury"])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)

# Prepare input data - match the feature order from training
numeric_mapping = {
    'CityTier': {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1},
    'Occupation': {'Salaried': 0, 'Freelancer': 1, 'Business': 2, 'Student': 3, 'Other': 4},
    'Gender': {'Male': 0, 'Female': 1},
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'Designation': {'Manager': 0, 'Senior Manager': 1, 'Director': 2, 'VP': 3, 'Entry Level': 4, 'Other': 5},
    'Passport': {'No': 0, 'Yes': 1},
    'OwnCar': {'No': 0, 'Yes': 1}
}

# Convert categorical values
city_tier_num = numeric_mapping['CityTier'][city_tier]
occupation_num = numeric_mapping['Occupation'][occupation]
gender_num = numeric_mapping['Gender'][gender]
marital_status_num = numeric_mapping['MaritalStatus'][marital_status]
designation_num = numeric_mapping['Designation'][designation]
passport_num = numeric_mapping['Passport'][passport]
own_car_num = numeric_mapping['OwnCar'][own_car]

# Assemble input into DataFrame with numeric values
input_data = pd.DataFrame([{
    # Numeric features
    'Age': age,
    'CityTier': city_tier_num,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfFollowups': number_of_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': passport_num,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car_num,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'MonthlyIncome': monthly_income,
    # Categorical features (will be one-hot encoded by the preprocessor in the pipeline)
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation
}])

# Predict button
if st.button("Predict Purchase"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.success("**Customer WILL purchase the Wellness Tourism Package!**")
            st.info(f"Purchase Probability: **{probability:.2%}**")
        else:
            st.warning("**Customer will NOT purchase the Wellness Tourism Package.**")
            st.info(f"Purchase Probability: **{probability:.2%}**")
        
        st.write("---")
        st.write("### Customer Summary")
        st.write(f"- Age: {age}")
        st.write(f"- City Tier: {city_tier}")
        st.write(f"- Monthly Income: ${monthly_income:,.2f}")
        st.write(f"- Number of Trips/Year: {number_of_trips}")
        st.write(f"- Pitch Satisfaction Score: {pitch_satisfaction_score}")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Make sure the input data matches the format the model was trained on.")
