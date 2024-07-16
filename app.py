import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model_path = os.path.join("artifact", "model.pkl")
    preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# Streamlit app
st.title('Credit Risk Prediction')

# Add a sidebar for dataset information
st.sidebar.header('Dataset Information')
with st.sidebar.expander("Status"):
    st.markdown("""
    - 1: No checking Account
    - 2: ... < 0 DM
    - 3: 0 <= ... < 200 DM
    - 4: ... >= 200 DM / salary for at least 1 year
    """)

with st.sidebar.expander("Purpose"):
    st.markdown("""
    - 0: Others
    - 1: Car (new)
    - 2: Car (used)
    - 3: Furniture/equipment
    - 4: Radio/television
    - 5: Domestic appliances
    - 6: Repairs
    - 7: Education
    - 8: Vacation
    - 9: Retraining
    - 10: Business
    """)

with st.sidebar.expander("Employment Duration"):
    st.markdown("""
    - 1: Unemployed
    - 2: < 1 yr
    - 3: 1 <= ... < 4 yrs
    - 4: 4 <= ... < 7 yrs
    - 5: >= 7 yrs
    """)

with st.sidebar.expander("Installment Rate"):
    st.markdown("""
    - 1: >= 35
    - 2: 25 <= ... < 35
    - 3: 20 <= ... < 25
    - 4: < 20
    """)

with st.sidebar.expander("Credit History"):
    st.markdown("""
    - 0: Delay in paying off in the past
    - 1: Critical account/other credits elsewhere
    - 2: No credits taken/all credits paid back duly
    - 3: Existing credits paid back duly till now
    - 4: All credits at this bank paid back duly
    """)

with st.sidebar.expander("Savings"):
    st.markdown("""
    - 1: Unknown/no savings account
    - 2: ... <  100 DM
    - 3: 100 <= ... <  500 DM
    - 4: 500 <= ... < 1000 DM
    - 5: ... >= 1000 DM
    """)


with st.sidebar.expander("Personal Status Sex"):
    st.markdown("""
    - 1: Male : divorced/separated
    - 2: Female : non-single or male : single
    - 3: Male : married/widowed
    - 4: Female : single
    """)
with st.sidebar.expander("Other Debtors"):
    st.markdown("""
    - 1: None
    - 2: Co-applicant
    - 3: Guarantor
    """)

with st.sidebar.expander("Number of Credits"):
    st.markdown("""
    - 1: 1
    - 2: 2-3
    - 3: 4-5
    - 4: >= 6
    """)


with st.sidebar.expander("Present Residence"):
    st.markdown("""
    - 1: < 1 yr
    - 2: 1 <= ... < 4 yrs
    - 3: 4 <= ... < 7 yrs
    - 4: >= 7 yrs
    """)

with st.sidebar.expander("Other Installment Plans"):
    st.markdown("""
    - 1: Bank
    - 2: Stores
    - 3: None
    """)

with st.sidebar.expander("Job"):
    st.markdown("""
    - 1: Unemployed/unskilled - non-resident
    - 2: Unskilled - resident
    - 3: Skilled employee/official
    - 4: Manager/self-empl./highly qualif. employee
    """)

with st.sidebar.expander("Property"):
    st.markdown("""
    - 1: Unknown / no property
    - 2: Car or other
    - 3: Building soc. savings agr./life insurance
    - 4: Real estate
    """)

with st.sidebar.expander("Housing"):
    st.markdown("""
    - 1: For free
    - 2: Rent
    - 3: Own
    """)


with st.sidebar.expander("People Liable"):
    st.markdown("""
    - 1: 3 or more
    - 2: 0 to 2
    """)
with st.sidebar.expander("Telephone"):
    st.markdown("""
    - 1: No
    - 2: Yes (under customer name)
    """)
with st.sidebar.expander("Foreign Worker"):
    st.markdown("""
    - 1: Yes
    - 2: No
    """)

# Organize input fields into sections
st.header('Personal Information')
col1, col2, col3 = st.columns(3)

with col1:
    status = st.selectbox('Status', options=[1, 2, 3, 4], help='Select the status of your checking account')
    duration = st.slider('Duration (months)', min_value=0, max_value=72, value=12, help='Duration of the credit in months')
    credit_history = st.selectbox('Credit History', options=[0, 1, 2, 3, 4], help='Select your credit history status')

with col2:
    purpose = st.selectbox('Purpose', options=list(range(11)), help='Purpose of the credit')
    amount = st.number_input('Amount', min_value=0, max_value=100000, value=5000, help='Credit amount')
    savings = st.selectbox('Savings', options=[1, 2, 3, 4, 5], help='Select your savings account status')

with col3:
    employment_duration = st.selectbox('Employment Duration', options=[1, 2, 3, 4, 5], help='Duration of your current employment')
    installment_rate = st.slider('Installment Rate (%)', min_value=1, max_value=4, value=1, help='Installment rate as percentage of disposable income')
    personal_status_sex = st.selectbox('Personal Status Sex', options=[1, 2, 3, 4], help='Select your personal status and sex')

st.header('Additional Information')
col4, col5, col6 = st.columns(3)

with col4:
    other_debtors = st.selectbox('Other Debtors', options=[1, 2, 3], help='Select other debtors/guarantors')
    present_residence = st.slider('Present Residence (years)', min_value=0, max_value=4, value=2, help='Number of years at present residence')
    property = st.selectbox('Property', options=[1, 2, 3, 4], help='Select the nature of your property')

with col5:
    age = st.number_input('Age', min_value=18, max_value=100, value=30, help='Enter your age')
    other_installment_plans = st.selectbox('Other Installment Plans', options=[1, 2, 3], help='Select other installment plans')
    housing = st.selectbox('Housing', options=[1, 2, 3], help='Select your housing status')

with col6:
    number_credits = st.slider('Number of Credits', min_value=1, max_value=4, value=1, help='Number of credits at this bank')
    job = st.selectbox('Job', options=[1, 2, 3, 4], help='Select your job category')
    people_liable = st.slider('People Liable', min_value=1, max_value=2, value=1, help='Number of people liable for maintenance')

st.header('Contact Information')
col7, col8 = st.columns(2)

with col7:
    telephone = st.selectbox('Telephone', options=[1, 2], help='Do you have a registered telephone?')
    foreign_worker = st.selectbox('Foreign Worker', options=[1, 2], help='Are you a foreign worker?')

# Button to make prediction
if st.button('Predict Credit Risk'):
    # Create a dictionary with all inputs
    input_data = {
        'status': status,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'amount': amount,
        'savings': savings,
        'employment_duration': employment_duration,
        'installment_rate': installment_rate,
        'personal_status_sex': personal_status_sex,
        'other_debtors': other_debtors,
        'present_residence': present_residence,
        'property': property,
        'age': age,
        'other_installment_plans': other_installment_plans,
        'housing': housing,
        'number_credits': number_credits,
        'job': job,
        'people_liable': people_liable,
        'telephone': telephone,
        'foreign_worker': foreign_worker
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Display result
    if prediction[0] == 1:
        st.success('Prediction: Safe')
    else:
        st.error('Prediction: Not Safe')

    # Display feature importances if available
    if hasattr(model, 'feature_importances_'):
        st.subheader('Feature Importances')
        feature_imp = pd.DataFrame({'feature': input_df.columns, 'importance': model.feature_importances_})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        st.bar_chart(feature_imp.set_index('feature'))

# Add some information about the project
st.sidebar.header('About')
st.sidebar.info('This app predicts credit risk based on various factors. It uses a machine learning model trained on historical data.')

# Add instructions
st.sidebar.header('Instructions')
st.sidebar.info('1. Enter the required information in the fields on the left.\n'
                '2. Click on the "Predict Credit Risk" button to see the prediction.\n'
                '3. The result will be displayed as either "Safe" or "Not Safe".')
