
import os
os.environ['NUMEXPR_MAX_THREADS'] = '1'

# rest of your code

import streamlit as st
import joblib
#import cloudpickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# with joblib.parallel_backend('loky', inner_max_num_threads=1):
#     ensemble_model_2 = joblib.load('stack_model_ensemble_model_2.joblib', mmap_mode='r')

ensemble_model_2 = joblib.load('super_learner_ensemble_model.joblib') 

# Add sidebar for page navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.selectbox("Select a page", ["Home", "Predict Attrition"])

# Define a function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def CreateNewFeatures(dataframe):
    dataframe["Age_risk"] = (dataframe["Age"] < 34).astype(int)
    dataframe["YearsAtCo_risk"] = (dataframe["YearsAtCompany"] < 4).astype(int)

    dataframe['NumCompaniesWorked'] = dataframe['NumCompaniesWorked'].replace(0, 1)
    dataframe['AverageTenure'] = dataframe["TotalWorkingYears"] / dataframe["NumCompaniesWorked"]

    dataframe['JobHopper'] = ((dataframe["NumCompaniesWorked"] > 2) & (dataframe["AverageTenure"] < 2.0)).astype(int)
 
    dataframe['feature_1'] = np.where(((dataframe['StockOptionLevel'] >= 1) & 
                                (dataframe['YearsAtCompany'] >= 3) & 
                                (dataframe['YearsWithCurrManager'] >= 1)), 1, 0)
    dataframe.drop(['YearsWithCurrManager'], axis=1, inplace=True)

    cols_to_scale = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "TotalWorkingYears", 
                     "YearsAtCompany", "AverageTenure", "YearsSinceLastPromotion"]
    scaler = StandardScaler()
    dataframe[cols_to_scale] = scaler.fit_transform(dataframe[cols_to_scale])

    return dataframe

def DropFeatures(dataframe):
    cat_features = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
                    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
                    'RelationshipSatisfaction', 'StockOptionLevel', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsSinceLastPromotion']
    
    ord_enc = OrdinalEncoder()
    ord_enc.fit(dataframe[cat_features])
    dataframe[cat_features] = ord_enc.transform(dataframe[cat_features])
    
    return dataframe


def Preprocessing(dataframe):
    dataframe = CreateNewFeatures(dataframe)
    dataframe = DropFeatures(dataframe)

    return dataframe


# Define the app
def app():
    st.title("Employee Attrition Prediction")

    st.write('Enter customer information to predict attrition')
    st.markdown('## Select features')

    age = st.number_input('Age', min_value=18, max_value=60, value=18) 
    performance_rating = st.selectbox('Performance Rating', [1, 2, 3, 4, 5])
    over_time = st.selectbox('Over Time', ['Yes', 'No'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    martial_status = st.selectbox('Martial Status', ['Single', 'Married', 'Divorced'])
    relationship_satisfaction = st.selectbox('Relationship Satisfaction', [1, 2, 3, 4])
    business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    distance_from_home = st.number_input('Distance From Home', min_value=1, max_value=1000, value=1)
    education_field = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])
    education = st.number_input('Education',min_value=1, max_value=30, value=1)
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    work_life_balance = st.selectbox('Work Life Balance', [1, 2, 3, 4])
    job_involvement = st.selectbox('Job Involvement', [0, 1, 2, 3, 4])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4])
    environment_satisfaction = st.selectbox('Environment Satisfaction', [0, 1, 2, 3, 4])
    stock_option_level = st.selectbox('Stock Option Level', [0, 1, 2, 3, 4])
    job_level = st.selectbox('Job Level', [1, 2, 3, 4, 5, 6, 7])
    training_times_last_year = st.selectbox('Training Times Last Year', [0, 1, 2, 3, 4, 5, 6])
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                                         'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources', 
                                         'Technical Leader'])
    num_companies_worked = st.number_input('Number of Companies Worked', min_value=1, max_value=20, value=1)
    percent_salary_hike = st.number_input('Percent Salary Hike', min_value=10, max_value=50, value=10)
    years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=50, value=0)
    years_with_current_manager = st.number_input('Years With Current Manager', min_value=0, max_value=50, value=0)
    years_at_company = st.number_input('Years At Company', min_value=1, max_value=50, value=1)
    total_working_years = st.number_input('Total Working Years', min_value=1, max_value=50, value=1)
    hourly_rate = st.number_input('Hourly Rate', min_value=20, max_value=100, value=20)
    daily_rate = st.number_input('Daily Rate', min_value=100, max_value=3000, value=100)
    monthly_rate = st.number_input('Monthly Rate', min_value=4000, max_value=30000, value=4000)

    if st.button('Predict Churn'):
        

        data = {
            'Age' : age,
            'Gender':gender,
            'PerformanceRating': performance_rating,
            'OverTime': over_time,
            'MaritalStatus': martial_status,
            'Department': department,
            'BusinessTravel': business_travel,
            'WorkLifeBalance': work_life_balance,
            'RelationshipSatisfaction': relationship_satisfaction,
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'StockOptionLevel': stock_option_level,
            'JobInvolvement': job_involvement,
            'EducationField': education_field,
            'Education':education,
            'JobLevel': job_level,
            'TrainingTimesLastYear': training_times_last_year,
            'JobRole': job_role,
            'NumCompaniesWorked':num_companies_worked,
            'PercentSalaryHike': percent_salary_hike,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'DistanceFromHome': distance_from_home,
            'YearsAtCompany': years_at_company,
            'TotalWorkingYears': total_working_years,
            'HourlyRate': hourly_rate,
            'DailyRate': daily_rate,
            'MonthlyRate': monthly_rate,
            'YearsWithCurrManager' : years_with_current_manager
            }
       
        temp_data = [data, data , data, data, data, data, data, data, data, data, data, data, data]
        features_df = pd.DataFrame.from_dict(temp_data)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write(features_df.head(1))


        #Preprocess inputs
        preprocess_df = Preprocessing(features_df)
        print(preprocess_df.shape)
        prediction = ensemble_model_2.predict(preprocess_df)
        

        prediction = pd.DataFrame(prediction)
    
        new_prediction = prediction.iloc[0]

        # Rename the columns
        new_prediction.columns = ['Attrition']
        st.write(new_prediction)

        if new_prediction.iloc[0] == 1:
            st.warning('Yes, the emlpoyee will leave the company.')
        else:
            st.success('No, the emlpoyee will stay in the company.')

        preprocess_df['Attrition'] = new_prediction
        preprocess_df["AttritionRisk"] = preprocess_df["Age_risk"] + preprocess_df["YearsAtCo_risk"] + preprocess_df['JobHopper']

        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk']
        risk_bins = [-1, 0, 1, 2, 3]
        preprocess_df['Risk_level'] = pd.cut(preprocess_df['AttritionRisk'], bins=risk_bins, labels=risk_labels, include_lowest=True)

        retention_plans = {'Low Risk': '•Conduct regular employee satisfaction surveys to identify and address any concerns or issues that may arise.'
        '•Provide opportunities for career development and growth within the company.'
        '•Offer competitive salaries and benefits to retain top talent.'
        '•Create a positive company culture that values and rewards employee contributions.',
                        
        'Medium Risk': '•Provide regular feedback and performance evaluations to keep employees engaged and motivated.'
        '•Offer opportunities for training and development to help employees grow their skills.'
        '•Provide a supportive work environment and flexible work arrangements.'
        '•Implement recognition and reward programs to acknowledge and appreciate employee contributions.',
                
        'High Risk': '•Conduct regular one-on-one meetings with employees to understand their needs and concerns.'
        '•Provide challenging and meaningful work assignments to keep employees engaged and motivated.'
        '•Offer opportunities for advancement and career growth within the company.'
        '•Provide competitive compensation and benefits packages to retain top talent.', 
                        
        'Extreme Risk': '•Conduct exit interviews to understand the reasons why employees are leaving and to identify areas for improvement.'
        '•Offer incentives and bonuses to retain key employees.'
        '•Implement mentorship or coaching programs to help employees stay engaged and motivated.'
        '•Provide opportunities for professional development and career advancement.'}

        preprocess_df['retention_plan'] = preprocess_df['Risk_level'].map(retention_plans)

        st.write(preprocess_df['Risk_level'].iloc[0])
            
        st.write(preprocess_df['retention_plan'].iloc[0])
app()
    
