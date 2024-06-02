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
import folium
import streamlit_folium as sf

# Function to load the trained models
@st.cache_data
def load_models():
    # Load your trained models here
    model_classification = joblib.load("model_classification_xgboost.pkl")
    model_sentence = joblib.load("model_sentence.pkl")
    model_cluster_0 = joblib.load('model_cluster_0_v2.pkl')
    scaler_cluster_0 = joblib.load('scaler_cluster_0.pkl')

    model_cluster_1 = joblib.load('model_cluster_1.pkl')
    model_cluster_2 = joblib.load('model_cluster_2.pkl')
    scaler_cluster_1 = joblib.load('scaler_cluster_1.pkl')
    scaler_cluster_2 = joblib.load('scaler_cluster_2.pkl')

    shoreline_data = gpd.read_file("full_water.geojson")
    shoreline_data = shoreline_data[shoreline_data['SUBSET'] == 'Bigwater waterbody']  # Filter for the

    # Reproject the shoreline data to EPSG:2926 (meters)
    shoreline_data = shoreline_data.to_crs("EPSG:4326")
    return model_classification, model_sentence, model_cluster_0, scaler_cluster_0, model_cluster_1, model_cluster_2, scaler_cluster_1, scaler_cluster_2, shoreline_data

@st.cache_data
def load_dfs():
    df_original = pd.read_csv('cleaned_data.csv')
    df_clustered = pd.read_csv("clustered_data_v2.csv")
    return df_original, df_clustered

@st.cache_data
def compile_text(bathroom_category, is_near_shore):
    text =  f"""Bathroom Category: {bathroom_category}, 
                Near Shore: {is_near_shore}, 
            """
    return text

@st.cache_data
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


def calculate_distance_shoreline_point_vs_coordinates(lat, long, closest_point):
    """
    Calculate the distance between two points using geodesic distance formula.

    Args:
        row (pandas.Series): A row containing latitude, longitude, and a point object.

    Returns:
        float: The distance between the two points in kilometers.
    """
    return geopy.distance.geodesic((lat, long), (closest_point.y, closest_point.x)).kilometers


def calculate_distance_to_shoreline_v2(lat,long,shoreline_data):
    # Load the shoreline data into a GeoDataFrame

    # Function to calculate distance between point and closest shoreline boundary
    def calculate_distance(lat,long):
        # Create a Point object with the latitude and longitude of the row
        point = Point(long, lat)
        # Find the closest polygon to the point
        closest_polygon = shoreline_data.geometry.distance(point).idxmin()
        # Get the closest polygon
        closest_polygon_geom = shoreline_data.geometry[closest_polygon]
        # Get the boundary of the closest polygon
        boundary = closest_polygon_geom.boundary
        # Find the closest point on the boundary to the given point
        closest_point_on_boundary = boundary.interpolate(boundary.project(point))
        # Calculate the distance to the closest point on the boundary
        closest_distance = point.distance(closest_point_on_boundary)
        return closest_distance, closest_point_on_boundary

    # Calculate distance for the first 20 rows in the DataFrame
    distance_to_shore, closest_point = calculate_distance(lat,long)
    distance_to_point_km = calculate_distance_shoreline_point_vs_coordinates(lat, long,closest_point)

    # generate map data
    # Create a folium map centered around the given location
    m = folium.Map(location=[lat, long], zoom_start=13)

    # Add marker for the given location
    folium.Marker([lat, long], popup='Your Location').add_to(m)

    # Add marker for the closest shore point
    folium.Marker([closest_point.y, closest_point.x], popup='Closest Shore Point').add_to(m)

    # Add line connecting the given location to the closest shore point
    folium.PolyLine([(lat, long), (closest_point.y, closest_point.x)], color="red", weight=2.5, opacity=1).add_to(m)

    return distance_to_point_km, m

@st.cache_data
def write_prediction_gap(prediction, rmse):
    # Calculate the price range based on RMSE
    price_lower_bound = prediction - rmse
    price_upper_bound = prediction + rmse

    # Round to the nearest 10,000
    price_lower_bound_rounded = round(price_lower_bound, -4)
    price_upper_bound_rounded = round(price_upper_bound, -4)

    # Format the prices with dot as thousand separator
    formatted_price_lower = f"{price_lower_bound_rounded:,}".replace(",", ".")
    formatted_price_upper = f"{price_upper_bound_rounded:,}".replace(",", ".")

    # Display the price range
    st.markdown(f"<h1 style='font-size:35px;'>House Price: ${formatted_price_lower} - ${formatted_price_upper}</h1>", unsafe_allow_html=True)

@st.cache_data
def fetch_address(street, city, province, country):
        if street == None:
            return None
    # Find Lat and Long of the address
        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(street+", "+city+", "+province+", "+country)

        if location is None:
            return None, None, None, None, None, None, None

        lat = location.latitude
        long = location.longitude

        # Calculate the distance to the nearest train station
        distance_to_station = calculate_distance_to_nearest_station(lat, long)
        distance_to_station = int(float(distance_to_station))

        # Get zipcode and additional information
        zipcode = get_zipcode(lat, long)
        
        population, density, poverty_value, commute_time, white_percentage = parse_zip_info(zipcode)
        if commute_time is None:
            return lat, long, distance_to_station, zipcode, None, None, None,location
        # Clean Data
        density = extract_density(density)
        population = int(population.replace(',', ''))
        commute_time = int(float(commute_time))

        return lat, long, distance_to_station, zipcode, population, density, commute_time, location


def main():
    # Load the models
    model_classification, model_sentence, model_cluster_0, scaler_cluster_0, model_cluster_1, model_cluster_2, scaler_cluster_1, scaler_cluster_2, shoreline_data  = load_models()

    df_original, df_clustered = load_dfs()

    # Title of the app
    st.title("House Price Prediction App")

    # Option to select the page to be displayed
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Choose the app mode", "Select Example", "Enter your House"])

    if app_mode == "Select Example":
        st.subheader("Selected Row")
        # Select an example
        row_number = st.sidebar.number_input("Row Number", min_value=0, max_value=len(df_clustered), value=0, step=1)
        if row_number != -1:
            # Display the selected example
            st.write(df_original.iloc[[row_number]])
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
            cluster_row = df_clustered.iloc[row_number]["cluster_all_data"]
            distance_to_point_km = df_clustered.iloc[row_number]["distance_to_point_km"]
            street = None
            city = None
            province = None
            country = None

        
    elif app_mode == "Enter your House":
        st.sidebar.header("Enter House Features")

        country = "United States"
        street = st.sidebar.text_input("Street", "2131 S Pearl St")
        city = st.sidebar.text_input("City", "Seattle")
        province = st.sidebar.text_input("Province", "Washington")
        st.sidebar.write("Country: " + country)

        # Sidebar inputs with default values
        input_grade = st.sidebar.slider("Grade of the House", min_value=0, max_value=13, value=10)  # Default is 10
        input_sqft_living = st.sidebar.number_input("Sqft Living", min_value=0, max_value=2000000, value=1000)  # Default is 1000
        input_bathroom_count = st.sidebar.slider("Bathroom Count", min_value=0, max_value=50, value=2)  # Default is 2
        input_seaside = st.sidebar.selectbox("Is Waterside:", ["Select an option", "Yes", "No"], index=0)  # Default is "Select an option"

    if app_mode == "Choose the app mode":
        input_grade = 0

    # Check if inputs are empty
    if input_grade == 0 or input_sqft_living == 0 or input_seaside == "Select an option":
        st.warning("Please fill in all the fields.")

    else:
        address = fetch_address(street, city, province, country)
        if  app_mode == "Enter your House" and address[0] is None:
            st.write("Address not found. Please check the address.")
        else:
            if app_mode == "Enter your House":
                lat, long, distance_to_station, zipcode, population, density, commute_time, location = fetch_address(street, city, province, country)
                if commute_time is None:
                    st.write("Cannot parse additional info. Please try again later")
                st.write(f"Zipcode: {zipcode}")
                st.write("Location: ", location)
                distance_to_point_km, m = calculate_distance_to_shoreline_v2(lat,long,shoreline_data)
                
        # Convert input features to a format suitable for prediction
            cluster_features = [input_bathroom_count, input_seaside] 

            # Make predictions based on user input
            with st.spinner('Calculating...'):
                time.sleep(1)  # Simulating calculation time
                if app_mode == "Enter your House":
                    house_cluster_prediction = predict_cluster(model_classification, model_sentence, cluster_features)[0]
                else:
                    house_cluster_prediction = cluster_row

            # Display the predictions
            st.subheader("Address of the House:")

            map_data = pd.DataFrame({'lat': [lat], 'lon': [long]})

            
            st.map(map_data,zoom=12) 
            st.write("Latitude:", lat)
            st.write("Longitude:", long)

            st.subheader("House Cluster Prediction:")
            st.write(f"House Cluster is predicted as ---> {house_cluster_prediction}")
            if app_mode == "Select Example":
                st.write("Actual House Cluster ---> ", df_clustered.iloc[row_number]["cluster_all_data"])
            
            st.subheader("Additional Info:")
            #st.write(f"Distance to the nearest train station: {distance_to_station} meters")

            st.write(f"Population: {population}")
            st.write(f"Density: {density}")
            st.write(f"Commute Time: {commute_time}")

            if app_mode == "Enter your House":
                st.write(f"Distance to the nearest shoreline is found as: {distance_to_point_km} km")
                sf.folium_static(m)

            # Predict the price of house:
            grade_living = input_grade*input_sqft_living

            grade_living_normalized = np.log1p(grade_living)

            if house_cluster_prediction == 0 :
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
                rmse_cluster = 110093

            elif house_cluster_prediction == 1:
                cluster_1_inputs = {
                    'grade_living_normalized': [grade_living_normalized],
                    'lat': [lat],
                    'commute_time': [commute_time],
                    'distance_to_point_km': [distance_to_point_km],
                    'nearest_station_distance_km': [distance_to_station]
                }
                cluster_1_inputs_df = pd.DataFrame(cluster_1_inputs)
                scaled_inputs = scaler_cluster_1.transform(cluster_1_inputs_df)
                predicted_price = model_cluster_1.predict(scaled_inputs)
                rmse_cluster = 75277

            elif house_cluster_prediction == 2:
                cluster_2_inputs = {
                    'grade_living_normalized': [grade_living_normalized],
                    'distance_to_point_km': [distance_to_point_km],
                    'nearest_station_distance_km': [distance_to_station],
                    'commute_time': [commute_time],
                    'lat': [lat]
                }

                cluster_2_inputs_df = pd.DataFrame(cluster_2_inputs)
                scaled_inputs = scaler_cluster_2.transform(cluster_2_inputs_df)
                predicted_price = model_cluster_2.predict(scaled_inputs)
                rmse_cluster = 256833

            else:
                st.write("Cluster not found. Error in prediction.")

            formatted_price = f"{int(predicted_price[0]):,}".replace(",", ".")
            st.markdown(f"<h1 style='font-size:48px;'>House Price: ${formatted_price}</h1>", unsafe_allow_html=True)

            if app_mode == "Select Example":
                # Retrieve the house price
                house_price = int(df_original.iloc[row_number]["price"])

                # Format the price with dot as thousand separator
                formatted_price = f"{house_price:,}".replace(",", ".")

                # Display formatted and larger price
                st.markdown(f"<h3 style='font-size:30px;'>Actual Price: ${formatted_price}</h1>", unsafe_allow_html=True)
                            
                st.write("Is record outlier: ", df_clustered.iloc[row_number]["outliers_ecod"])

            
            write_prediction_gap(prediction=int(predicted_price) , rmse=rmse_cluster)

if __name__ == "__main__":
    main()
