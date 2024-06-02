import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib

# Veriyi yükleme ve işleme (Önceki adımlar)
df = pd.read_csv('clustered_data_v2.csv')
important_features = ['grade_living_normalized', 'distance_to_point_km', 'nearest_station_distance_km', 'commute_time', 'lat','cluster_all_data','price']
df_wo_main = df[important_features]

# IQR Yöntemi ile Aykırı Değerleri Bulma ve Baskılamak
def cap_outliers(df):
    capped_df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        capped_df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        capped_df[col] = np.where(df[col] > upper_bound, upper_bound, capped_df[col])
    return capped_df

df_wo_main_capped = cap_outliers(df_wo_main)


# Modeli eğitme fonksiyonu
def train_catboost_model(df):
    y = df[df['cluster_all_data'] == 2]['price']
    X = df[df['cluster_all_data'] == 2].drop(columns=['cluster_all_data', 'price'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # StandardScaler ile normalizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CatBoostRegressor(verbose=0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, 'model_cluster_2.pkl')
    joblib.dump(scaler, 'scaler_cluster_2.pkl')

train_catboost_model(df_wo_main_capped)

