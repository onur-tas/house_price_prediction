import streamlit as st
import joblib
import time

# Function to load the trained models
@st.cache_data
def load_models():
    # Load your trained models here
    model_classification = joblib.load("model_classification_xgboost.pkl")
    model_sentence = joblib.load("model_sentence.pkl")
    return model_classification, model_sentence

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

def main():
    # Load the models
    model_classification, model_sentence = load_models()

    # Title of the app
    st.title("House Price Prediction App")

    # Input form for user to enter house features
    st.sidebar.header("Enter House Features")
    # Example input fields, replace with your actual input fields
    input_grade = st.sidebar.slider("Grade of the House", min_value=0, max_value=13, value=0)
    input_sqft_living = st.sidebar.number_input("Sqft Living", min_value=0, max_value=2000000, value=0)
    input_bathroom_count = st.sidebar.slider("Bathroom Count", min_value=0, max_value=50, value=0)
    input_seaside  = st.sidebar.selectbox("Is Waterside:", ["Select an option", "Yes", "No"])


    # Check if inputs are empty
    if input_grade == 0 or input_sqft_living == 0 or input_seaside == "Select an option":
        st.warning("Please fill in all the fields.")
    else:
        # Convert input features to a format suitable for prediction
        cluster_features = [input_bathroom_count, input_seaside ] 

        # Make predictions based on user input
        with st.spinner('Calculating...'):
            time.sleep(2)  # Simulating calculation time
            house_cluster_prediction = predict_cluster(model_classification, model_sentence, cluster_features)

        # Display the predictions
        st.subheader("Predictions:")
        st.write(f"House Cluster: {house_cluster_prediction}")
        st.write("House Price: $x.xx")  # Placeholder for actual price prediction

if __name__ == "__main__":
    main()
