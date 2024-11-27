import numpy as np
import streamlit as st
import pickle
from xgboost import XGBClassifier
from PIL import Image

# Load encoders and model
def load_encoding():
    with open(r'model/checkpoint.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_encoding()

xgb = XGBClassifier()
xgb.load_model(r'model/model_xgb.bin')

# Encoders
le_Day_of_week = data['le_Day_of_week']
le_Vehicle_driver_relation = data['le_Vehicle_driver_relation']
le_Road_surface_type = data['le_Road_surface_type']
le_Road_surface_conditions = data['le_Road_surface_conditions']
le_Type_of_collision = data['le_Type_of_collision']
le_Vehicle_movement = data['le_Vehicle_movement']
le_Work_of_casuality = data['le_Work_of_casuality']
le_Cause_of_accident = data['le_Cause_of_accident']

# Custom dictionaries for mapping
Time_dict = {"Day": 0, "Night": 1}
Age_band_of_driver_dict = {"Under 18": 0, "18-30": 1, "31-50": 2, "Over 51": 3}
Driving_experience_dict = {"Below 1yr": 0, "1-2yr": 1, "2-5yr": 2, "5-10yr": 3, "Above 10yr": 4, "No Licence": 5}
Service_year_of_vehicle_dict = {"Below 1yr": 0, "1-2yr": 1, "2-5yrs": 2, "5-10yrs": 3, "Above 10yr": 4}
Light_conditions_dict = {"Daylight": 0, "Darkness - lights lit": 1, "Darkness - no lighting": 2}
Age_band_of_casualty_dict = {"Under 18": 0, "18-30": 1, "31-50": 2, "Over 51": 3}

st.set_page_config(page_title="Accident Severity Prediction App 🚧",
                   page_icon="🚦", layout="wide")

# Header with image
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    image = Image.open('doc/theme.png')
    st.image(image, use_container_width=True)
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Accident Severity Prediction 🚦</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Predict the severity of road accidents based on input parameters.</p>", unsafe_allow_html=True)

# Custom styling for inputs
st.markdown("""<style>
    .stSlider > div {background-color: #f0f0f0;}
    .stButton button {background-color: #FF5733; color: white; border-radius: 8px;}
    .stSelectbox {background-color: #f9f9f9;}
</style>""", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):
        st.subheader("Enter the details below:")

        # Use columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            time = st.selectbox("Select Time of Day:", options=["Day", "Night"])
            day_of_week = st.selectbox("Day of the Week:", options=['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            driver_age = st.selectbox("Driver's Age Group:", options=['18-30', '31-50', 'Over 51', 'Under 18'])
            vehicle_relation = st.selectbox("Vehicle Driver Relation:", options=['Employee', 'Owner'])
            driving_experience = st.selectbox("Driving Experience:", options=['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence'])

        with col2:
            service_year_of_vehicle = st.selectbox("Service Year of Vehicle:", options=['Below 1yr', '1-2yr', '2-5yrs', '5-10yrs', 'Above 10yr'])
            road_surface_type = st.selectbox("Road Surface Type:", options=['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress', 'Gravel roads', 'other'])
            road_surface_conditions = st.selectbox("Road Surface Conditions:", options=['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'])
            light_conditions = st.selectbox("Light Conditions:", options=['Darkness - no lighting', 'Darkness - lights lit', 'Daylight'])
            type_of_collision = st.selectbox("Type of Collision:", options=['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with animals', 'Rollover', 'Fall from vehicles', 'Collision with pedestrians', 'With Train'])

        col3, col4 = st.columns(2)
        with col3:
            number_of_vehicles_involved = st.slider("Number of Vehicles Involved:", 1, 7, value=1)
        with col4:
            number_of_casualties = st.slider("Number of Casualties:", 1, 8, value=1)

        vehicle_movement = st.selectbox("Vehicle Movement:", options=['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go', 'Getting off', 'Reversing', 'Parked', 'Stopping', 'Overtaking', 'Entering a junction'])
        age_band_of_casualty = st.selectbox("Casualty's Age Group:", options=['Under 18', '18-30', '31-50', 'Over 51'])
        work_of_casuality = st.selectbox("Work of Casualty:", options=['Driver', 'Employee', 'Self-employed', 'Student', 'Unemployed'])
        cause_of_accident = st.selectbox("Cause of Accident:", options=['Moving Backward', 'Overtaking', 'Changing lane to the left', 'Changing lane to the right', 'Overloading', 'No priority to vehicle', 'No priority to pedestrian', 'No distancing', 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed', 'Driving carelessly', 'Driving at high speed', 'Driving to the left', 'Overturning', 'Turnover', 'Driving under the influence of drugs', 'Drunk driving'])

        submit = st.form_submit_button("Predict")

    if submit:
        # Transform inputs for prediction
        time = Time_dict[time]
        day_of_week = le_Day_of_week.transform([day_of_week])
        driver_age = Age_band_of_driver_dict[driver_age]
        vehicle_relation = le_Vehicle_driver_relation.transform([vehicle_relation])
        driving_experience = Driving_experience_dict[driving_experience]
        service_year_of_vehicle = Service_year_of_vehicle_dict[service_year_of_vehicle]
        road_surface_type = le_Road_surface_type.transform([road_surface_type])
        road_surface_conditions = le_Road_surface_conditions.transform([road_surface_conditions])
        light_conditions = Light_conditions_dict[light_conditions]
        type_of_collision = le_Type_of_collision.transform([type_of_collision])
        vehicle_movement = le_Vehicle_movement.transform([vehicle_movement])
        age_band_of_casualty = Age_band_of_casualty_dict[age_band_of_casualty]
        work_of_casuality = le_Work_of_casuality.transform([work_of_casuality])
        cause_of_accident = le_Cause_of_accident.transform([cause_of_accident])

        user_inp = np.asarray([[time, *day_of_week, driver_age, *vehicle_relation, driving_experience,
                                service_year_of_vehicle, *road_surface_type, *road_surface_conditions,
                                light_conditions, *type_of_collision, number_of_vehicles_involved,
                                number_of_casualties, *vehicle_movement, age_band_of_casualty,
                                *work_of_casuality, *cause_of_accident]])

        # Dummy prediction logic
        pred = "Slight injury"  # Replace with get_prediction(xgb, user_inp)

        if pred == 'Slight injury':
            st.success('Thank God! It was a Slight Injury!')
        elif pred == 'Serious Injury':
            st.warning('It seems like Serious Injury!')
        else:
            st.error('OMG it\'s a Fatal Injury. Hope the driver recovers fast.')

    # Add a footer section with the message from Aditya
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>Greetings from <b>Aditya</b>. Thanks for using this application! 🚗</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
