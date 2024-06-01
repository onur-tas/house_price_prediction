import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle


data = pd.read_csv("./clustered_data_v2.csv")

data_corr_1 = data[data["cluster_all_data"] == 1]

data_model = data_corr_1[['price', 'grade_living_normalized', 'lat', 'commute_time', 'distance_to_point_km', 'nearest_station_distance_km']]

X = data_model.drop('price', axis=1) 
y = data_model['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def eval_function(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

xgb_model = XGBRegressor()
xgb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5)
xgb_grid_search.fit(X_train, y_train)
xgb_best_model = xgb_grid_search.best_estimator_
xgb_pred_tuned = xgb_best_model.predict(X_test)
xgb_scores_tuned = eval_function(y_test, xgb_pred_tuned)

with open('./xgb_best_model_cluster1.pkl', 'wb') as file:
    pickle.dump(xgb_best_model, file)

with open('./scaler_cluster1.pkl', 'wb') as file:
    pickle.dump(scaler, file)


print("XGBoost Modeli (Tuned) - MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(*xgb_scores_tuned))
