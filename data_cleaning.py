import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json
import requests
from bs4 import BeautifulSoup
import re
import geopandas as gpd
from shapely.geometry import Point, LineString
import geopy
from shapely import wkt



def get_zipcode(latitude, longitude):
    """
    Retrieves the zipcode based on the given latitude and longitude coordinates.

    Parameters:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    str or None: The zipcode corresponding to the given coordinates. Returns None if the zipcode is not found.
    """
    geolocator = Nominatim(user_agent="zipcode_finder")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    address = location.raw['address']
    if 'postcode' in address:
        return address['postcode']
    else:
        return None

def find_missing_zip_code(df):
    """Finds missing zip codes in the given DataFrame using latitude and longitude coordinates.
    
    Args:
        df (DataFrame): The DataFrame containing the data. Make sure lat and long columns are present.
        
    Returns:
        None
    """
    for index, row in df[df['zipcode'].isna()].iterrows():
        latitude = row['lat']
        longitude = row['long']
        zipcode = get_zipcode(latitude, longitude)
        print('Zipcode found', zipcode)
        df.at[index, 'zipcode'] = zipcode

def extract_date_info(df):
    """
    Extracts date information from a DataFrame and adds new columns for year, month, quarter, and months since built.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the date column.
    
    Returns:
        None
    """
    df['timestamp'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['months_since_built'] = (df['year'] - df['yr_built']) * 12 + (df['month'] - 1)
    df.drop(columns=['timestamp'], inplace=True)

def calculate_distance_to_nearest_station(df):
    """
    Calculates the distance between each row in the DataFrame and the nearest train station.
    Creates a new row named "nearest_station_distance_km" with the calculated distances.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    None
    """
    # Load the GeoJSON file
    with open('metro_stations_seattle.geojson', 'r') as f:
        metro_data = json.load(f)

    # Extract coordinates of train stations
    station_coords = [(feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]) 
                    for feature in metro_data['features']]

    # Define a function to calculate the distance between two coordinates
    def calculate_distance(row):
        row_coord = (row['lat'], row['long'])
        # Calculate distances to all train stations
        distances = [geodesic(row_coord, station).meters for station in station_coords]
        # Find the minimum distance
        min_distance = min(distances)
        return min_distance

    # Apply the calculate_distance function to each row of the DataFrame
    df['nearest_station_distance_km'] = df.apply(calculate_distance, axis=1)

def create_bathroom_category(df):
    """
    Create a new column 'bathroom_category' in the DataFrame 'df' based on the number of bathrooms.
    Creates categories in column named 'bathroom_category'.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    None
    """

    bathroom_categories = {
        (0, 1): "Studio or 1 Bathroom",
        (1, 2): "1-2 Bathrooms",
        (2, 3): "2-3 Bathrooms",
        (3, float('inf')): "+3 Bathrooms",
    }

    # Create a function to assign categories based on the number of bathrooms
    def assign_bathroom_category(num_bathrooms):
        for category_range, category_label in bathroom_categories.items():
            if num_bathrooms > category_range[0] and num_bathrooms <= category_range[1]:
                return category_label

    # Assuming 'df' is your DataFrame and 'bathrooms' is the column containing the number of bathrooms
    df['bathroom_category'] = df['bathrooms'].apply(assign_bathroom_category)

def parse_zip_info(zipcode):
    """
    Parses the zip code information from a website and extracts population, density, poverty value, commute time, and white percentage.

    Args:
        zipcode (str): The zip code to retrieve information for.

    Returns:
        tuple: A tuple containing the population, density, poverty value, commute time, and white percentage.
               If the request fails, returns None for all values.
    """

    # URL for the website with the specified zip code
    url = f"https://simplemaps.com/us-zips/{zipcode}/"
    
    # Send an HTTP GET request to the website
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Request Successful for {}".format(zipcode))
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing population and density information
        table = soup.find("table", class_="table-condensed")
        
        # Extract population and density from the table
        population = None
        density = None
        if table:
            # Iterate over rows in the table
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) == 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip()
                    if label == "Population":
                        population = value
                    elif label == "Density":
                        density = value
        
        # Find the <span> elements containing the commute time, poverty, and white percentage values
        span_elements = soup.find_all("span", style="font-size: 46px")
        # Extract the commute time, poverty, and white percentage values
        commute_time_value = None
        poverty_value = None
        white_percentage_value = None
        for span_element in span_elements:
            span_text = span_element.text.strip()
            span_match = re.search(r'\d+\.\d+', span_text)
            if span_match:
                # Check if the value represents commute time, poverty, or white percentage
                if "minutes" in span_text:
                    commute_time_value = span_match.group()
                elif "Poverty" in span_text:
                    poverty_value = span_match.group()
                elif "White" in span_text:
                    white_percentage_value = span_match.group()
        
        # Return all extracted information
        return population, density, poverty_value, commute_time_value, white_percentage_value
    
    else:
        # Return None for all values if the request fails
        print("Request Failed for {}".format(zipcode))
        return None, None, None, None, None

def extract_density(text):
    """
    Extracts the density value from the given text.

    Args:
        text (str): The text containing the density value.

    Returns:
        int or None: The extracted density value rounded to the nearest integer,
        or None if no density value is found in the text.
    """
    match = re.search(r'(\d+(\.\d+)?)', text)
    if match:
        return round(float(match.group(1)))
    else:
        return None

def scrape_zipcode_additonal_data(df):
    """
    Scrapes additional data for each unique zip code in the DataFrame and updates the DataFrame with the scraped values.

    Args:
        df (pandas.DataFrame): The DataFrame containing the zip code data.

    Returns:
        None
    """

    # Convert zipcode to integer type
    df["zipcode"] = df["zipcode"].astype(int)

    # Iterate over unique zip codes in the DataFrame
    for zipcode in df["zipcode"].unique():
        # Parse information for the current zip code
        population, density, poverty_value, commute_time_value, white_percentage_value = parse_zip_info(zipcode)
        
        # Update DataFrame with scraped values for all rows with the current zip code
        df.loc[df["zipcode"] == zipcode, ["population", "density", "poverty_value", "commute_time", "white_percentage"]] = [
            population, extract_density(density), poverty_value, commute_time_value, white_percentage_value
        ]

    # Convert population to integer type
    df['population'] = df['population'].str.replace(',', '').astype(int)
    df['commute_time'] = pd.to_numeric(df['commute_time'], errors='coerce')
    df.drop(columns=['poverty_value','white_percentage'], inplace=True)

def calculate_distance_shoreline_point_vs_coordinates(row):
    """
    Calculate the distance between two points using geodesic distance formula.

    Args:
        row (pandas.Series): A row containing latitude, longitude, and a point object.

    Returns:
        float: The distance between the two points in kilometers.
    """
    return geopy.distance.geodesic((row['lat'], row['long']), (row['closest_point'].y, row['closest_point'].x)).kilometers

def calculate_distance_to_nearest_shore_polygon(row, shoreline_data):
    # Create a Point object with the latitude and longitude of the row
    point = Point(row['long'], row['lat'])
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
    # Convert the closest point to a WKT string
    closest_point_wkt = wkt.dumps(closest_point_on_boundary)
    return closest_distance, closest_point_wkt

def calculate_distance_to_shoreline(df):
    """
    Calculates the distance between each point in the DataFrame `df` and the closest shoreline boundary.

    Parameters:
    df (DataFrame): The DataFrame containing the points for which the distance needs to be calculated.

    Returns:
    None
    """
    # Load the shoreline data into a GeoDataFrame
    shoreline_data = gpd.read_file("full_water.geojson")
     # Filter for the 'Bigwater waterbody' subset
    shoreline_data = shoreline_data[shoreline_data['SUBSET'] == 'Bigwater waterbody'] 
    # Reproject the shoreline data to EPSG:4326
    shoreline_data = shoreline_data.to_crs("EPSG:4326")
    # Function to calculate distance between point and closest shoreline boundary
    df['distance_to_shore'], df['closest_point'] = zip(*df.apply(lambda row: calculate_distance_to_nearest_shore_polygon(row, shoreline_data), axis=1))
    # Convert the 'POINT' column to Shapely Point objects
    # Calculate distance between each pair of points in km
    

def calculate_distance_to_shoreline_v2(df):
    # Load the shoreline data into a GeoDataFrame
    shoreline_data = gpd.read_file("full_water.geojson")
    shoreline_data = shoreline_data[shoreline_data['SUBSET'] == 'Bigwater waterbody']  # Filter for the

    # Reproject the shoreline data to EPSG:2926 (meters)
    shoreline_data = shoreline_data.to_crs("EPSG:4326")

    # Function to calculate distance between point and closest shoreline boundary
    def calculate_distance(row):
        # Create a Point object with the latitude and longitude of the row
        point = Point(row['long'], row['lat'])
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
    df['distance_to_shore'], df['closest_point'] = zip(*df.apply(calculate_distance, axis=1))
    df['distance_to_point_km'] = df.apply(lambda row: calculate_distance_shoreline_point_vs_coordinates(row), axis=1)
    df['is_near_shore'] = df['distance_to_point_km'] < 0.5
    df.drop(columns=['distance_to_shore'], inplace=True)


if __name__ == "__main__":
    print('initiated')
