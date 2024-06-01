# Streamlit uygulaması
st.title('Ev Fiyatı Tahmin Uygulaması')

# Kullanıcıdan ev özelliklerini alma
grade_living_normalized = st.number_input('Grade Living Normalized', min_value=0.0, value=5.0)
distance_to_point_km = st.number_input('Distance to Point (km)', min_value=0.0, value=1.0)
nearest_station_distance_km = st.number_input('Nearest Station Distance (km)', min_value=0.0, value=1.0)
commute_time = st.number_input('Commute Time (minutes)', min_value=0, value=10)
lat = st.number_input('Latitude', min_value=0.0, value=40.0)

# Tahmin butonu
if st.button('Tahmin Et'):
    # Kaydedilen modeli ve scaler'ı yükleme
    model = joblib.load('catboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Kullanıcı girdilerini modele uygun formata getirme
    input_data = np.array([[grade_living_normalized, distance_to_point_km, nearest_station_distance_km, commute_time, lat]])
    input_data_scaled = scaler.transform(input_data)
    
    # Tahmin yapma
    prediction = model.predict(input_data_scaled)
    
    # Sonucu ekrana yazdırma
    st.write(f'Tahmin edilen ev fiyatı: {prediction[0]:,.2f} TL')

# Model performansını ve feature importances'i değerlendirme ve görselleştirme
if st.checkbox('Model Performansı ve Feature Importances'):
    # Performans ve feature importances hesaplama
    y_wo_main_capped = df_wo_main_capped[df_wo_main_capped['cluster_all_data'] == 2]['price']
    X_wo_main_capped = df_wo_main_capped[df_wo_main_capped['cluster_all_data'] == 2].drop(columns=['cluster_all_data', 'price'], axis=1)
    
    results, feature_importances, feature_names = evaluate_catboost_model_with_scaler(X_wo_main_capped, y_wo_main_capped)
    
    st.write("Results for CatBoost model with dataset without main columns (StandardScaler normalization):")
    st.write(f"CatBoost - MSE: {results['MSE']}, R2: {results['R2']}")

    # Feature importances grafiği
    def plot_feature_importances(feature_importances, feature_names, model_name, dataset_name):
        plt.figure(figsize=(10, 6))
        indices = np.argsort(feature_importances)[::-1]
        plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
        plt.xticks(range(len(feature_importances)), feature_names[indices], rotation=90)
        plt.title(f"Feature Importances for {model_name} ({dataset_name})")
        st.pyplot(plt)

    plot_feature_importances(feature_importances, feature_names, "CatBoost", "without main columns (StandardScaler normalization)")
