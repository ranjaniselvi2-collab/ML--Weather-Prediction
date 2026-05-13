# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset

The main objective of this project is to implement the Random Forest Algorithm for weather prediction. The dataset contains weather details such as temperature, humidity, and wind speed. Using these input features, the model predicts whether rain will occur or not. The Random Forest model is trained using the dataset and used to make accurate weather predictions

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create the weather dataset.

2. Select the input features and target variable from the dataset.

3. Create the Random Forest Classifier model.

4. Train the model using the given weather data.

5. Predict the weather result and display the prediction output.


## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Ranjani S
RegisterNumber: 212225230224

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = {
    'Humidity': [60, 65, 70, 75, 80, 85, 90, 55, 50, 45],
    'Pressure': [1012, 1010, 1008, 1005, 1003, 1001, 1000, 1015, 1018, 1020],
    'WindSpeed': [5, 6, 7, 5, 4, 3, 2, 6, 7, 8],
    'Temperature': [30, 29, 28, 27, 26, 25, 24, 31, 32, 33],
    'PM2.5': [80, 85, 90, 95, 100, 110, 120, 70, 65, 60],
    'Energy': [200, 210, 220, 230, 240, 250, 260, 190, 185, 180]
}

df = pd.DataFrame(data)


X = df[['Humidity', 'Pressure', 'WindSpeed']]
y = df[['Temperature', 'PM2.5', 'Energy']]  # multi-output regression


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross Validation R2 Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

*/
```

## Output:

<img width="1097" height="152" alt="Screenshot 2026-05-11 204756" src="https://github.com/user-attachments/assets/a8564fe9-2e4a-4e86-9569-d1333d1b8e63" />


## Result:

The Random Forest Algorithm successfully predicted daily temperature, PM2.5 pollution level, and energy consumption accurately using environmental sensor data.
