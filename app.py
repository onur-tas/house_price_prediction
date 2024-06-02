import streamlit as st
import joblib
import time
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import json
import numpy as np
from data_cleaning import get_zipcode, parse_zip_info, extract_density


# Function to load the trained models
#@st.cache_data
def load_models():
    # Load your trained models here
    model_classification = joblib.load("model_classification_xgboost.pkl")
    model_sentence = joblib.load("model_sentence.pkl")
    model_cluster_0 = joblib.load('model_cluster_0_v2.pkl')
    scaler_cluster_0 = joblib.load('scaler_cluster_0.pkl')
    return model_classification, model_sentence, model_cluster_0, scaler_cluster_0

#@st.cache_data
def load_dfs():
    df_original = pd.read_csv('cleaned_data.csv')
    df_clustered = pd.read_csv("clustered_data_v2.csv")
    return df_original, df_clustered

def compile_text(bathroom_category, is_near_shore):
    text =  f"""Bathroom Category: {bathroom_category}, 
                Near Shore: {is_near_shore}, 
            """
    return text

def categorize_bathrooms(num_bathrooms):
    if num_bathrooms == 0 or num_bathrooms == 1:
        return "Studio or 1 Bathroom"
    elif num_bathrooms < 3:
        return "1-2 Bathrooms"
    elif num_bathrooms < 4:
        return "2-3 Bathrooms"
    else:
        return "+3 Bathrooms"

# Function to make predictions using the loaded models
def predict_cluster(model_cluster, model_sentence, cluster_features):
    # Perform any necessary data preprocessing on input_features
    input_bathroom_count = cluster_features[0]
    input_seaside = True if cluster_features[1] == "Yes" else False
    bathroom_category = categorize_bathrooms(input_bathroom_count)
    
    compiled_text = compile_text(bathroom_category, input_seaside)
    compiled_text_list = [compiled_text]
    encoded_text = model_sentence.encode(sentences=compiled_text_list, show_progress_bar=True, normalize_embeddings=True)
    # Make predictions using the loaded models 
    prediction1 = model_cluster.predict(encoded_text)
    return prediction1


def calculate_distance_to_nearest_station(lat, long):
    """
    Calculates the distance between each row in the DataFrame and the nearest train station.
    Creates a new row named "nearest_station_distance_km" with the calculated distances.

    Parameters:
    lat (float): Latitude of the location
    long (float): Longitude of the location

    Returns:
    distance to the nearest train station in meters
    """
    # Load the GeoJSON file
    with open('metro_stations_seattle.geojson', 'r') as f:
        metro_data = json.load(f)

    # Extract coordinates of train stations
    station_coords = [(feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]) 
                    for feature in metro_data['features']]

    house_coord = (lat, long)
    # Calculate distances to all train stations
    distances = [geodesic(house_coord, station).meters for station in station_coords]
    # Find the minimum distance
    min_distance = min(distances)
    return min_distance

def main():
    # Load the models
    model_classification, model_sentence, model_cluster_0, scaler_cluster_0 = load_models()

    df_original, df_clustered = load_dfs()

    # Title of the app
    st.title("House Price Prediction App")

    # Option to select the page to be displayed
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Choose the app mode", "Select Example", "Enter your House"])

    if app_mode == "Select Example":
        st.subheader("Select Example")
        # Select an example
        row_number = st.sidebar.number_input("Row Number", min_value=0, max_value=len(df_clustered), value=0, step=1)
        if row_number != -1:
            # Display the selected example
            st.dataframe(data=df_original.iloc[row_number], width=900, height=200)
            input_grade = df_original.iloc[row_number]["grade"]
            input_sqft_living = df_original.iloc[row_number]["sqft_living"]
            input_bathroom_count = df_clustered.iloc[row_number]["bathrooms"]
            input_seaside = df_clustered.iloc[row_number]["waterfront"]
            lat = df_clustered.iloc[row_number]["lat"]
            long = df_clustered.iloc[row_number]["long"]
            distance_to_station = df_clustered.iloc[row_number]["nearest_station_distance_km"]
            zipcode = df_clustered.iloc[row_number]["zipcode"]
            population = df_original.iloc[row_number]["population"]
            density = df_original.iloc[row_number]["density"]
            commute_time = df_original.iloc[row_number]["commute_time"]
        
    elif app_mode == "Enter your House":
        st.sidebar.header("Enter House Features")

        country = "United States"
        street = st.sidebar.text_input("Street", "61st Ave S")
        city = st.sidebar.text_input("City", "Seattle")
        province = st.sidebar.text_input("Province", "Washington")
        st.sidebar.write("Country: " + country)

        # Sidebar inputs with default values
        input_grade = st.sidebar.slider("Grade of the House", min_value=0, max_value=13, value=10)  # Default is 10
        input_sqft_living = st.sidebar.number_input("Sqft Living", min_value=0, max_value=2000000, value=1000)  # Default is 1000
        input_bathroom_count = st.sidebar.slider("Bathroom Count", min_value=0, max_value=50, value=2)  # Default is 2
        input_seaside = st.sidebar.selectbox("Is Waterside:", ["Select an option", "Yes", "No"], index=0)  # Default is "Select an option"

        # Find Lat and Long of the address
        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(street+", "+city+", "+province+", "+country)

        lat = location.latitude
        long = location.longitude

        # Calculate the distance to the nearest train station
        distance_to_station = calculate_distance_to_nearest_station(lat, long)
        distance_to_station = int(float(distance_to_station))

        # Get zipcode and additional information
        zipcode = get_zipcode(lat, long)
        st.write(f"Zipcode: {zipcode}")
        population, density, poverty_value, commute_time, white_percentage = parse_zip_info(zipcode)
        # Clean Data
        density = extract_density(density)
        population = int(population.replace(',', ''))
        commute_time = int(float(commute_time))

        st.write("Location: ", location)


    if app_mode == "Choose the app mode":
        input_grade = 0

    # Check if inputs are empty
    if input_grade == 0 or input_sqft_living == 0 or input_seaside == "Select an option":
        st.warning("Please fill in all the fields.")

    else:
        # Convert input features to a format suitable for prediction
        cluster_features = [input_bathroom_count, input_seaside] 

        # Make predictions based on user input
        with st.spinner('Calculating...'):
            time.sleep(2)  # Simulating calculation time
            house_cluster_prediction = predict_cluster(model_classification, model_sentence, cluster_features)

        # Display the predictions
        st.subheader("Address of the House:")

        map_data = pd.DataFrame({'lat': [lat], 'lon': [long]})

        
        st.map(map_data,zoom=12) 
        st.write("Latitude:", lat)
        st.write("Longitude:", long)

        st.subheader("House Cluster Prediction:")
        st.write(f"House Cluster is predicted as ---> {house_cluster_prediction[0]}")
        if app_mode == "Select Example":
            st.write("Actual House Cluster ---> ", df_clustered.iloc[row_number]["cluster_all_data"])
        
        st.subheader("Additional Info:")
        st.write(f"Distance to the nearest train station: {distance_to_station} meters")

        st.write(f"Population: {population}")
        st.write(f"Density: {density}")
        st.write(f"Commute Time: {commute_time}")

        # Predict the price of house:
        grade_living = input_grade*input_sqft_living

        grade_living_normalized = np.log1p(grade_living)


        # create df from the inputs
        cluster_0_inputs = {
            'grade_living_normalized': [grade_living_normalized],
            'lat': [lat],
            'nearest_station_distance_km': [distance_to_station]
        }
        # Convert dictionary to DataFrame
        cluster_0_inputs_df = pd.DataFrame(cluster_0_inputs)
        scaled_inputs = scaler_cluster_0.transform(cluster_0_inputs_df)
        predicted_price = model_cluster_0.predict(scaled_inputs)

        # Placeholder for actual price prediction
        st.subheader(f"House Price Prediction:  {predicted_price[0]}" )
        st.write("House Price:", predicted_price[0])  # Placeholder for actual price prediction
        if app_mode == "Select Example":
            st.write("Actual House Price ---> ", df_original.iloc[row_number]["price"])

        if app_mode == "Select Example":
            st.write("Is record outlier: ", df_clustered.iloc[row_number]["outliers_ecod"])

if __name__ == "__main__":
    main()
