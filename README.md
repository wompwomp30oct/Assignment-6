# Assignment 6: Weather Condition Classification using SVM and Open-Meteo API

## Objective
Build an SVM-based weather classification model that predicts whether weather conditions are **Cool** or **Warm** using meteorological observations fetched from the Open-Meteo API.

## API Documentation
- Open-Meteo API: https://open-meteo.com/

## Libraries Used
- `pandas`
- `requests`
- `scikit-learn`

## Methodology
1. Fetched hourly weather data from the Open-Meteo API.
2. Combined observations from multiple locations to ensure both weather classes were present.
3. Converted the JSON response into a Pandas DataFrame.
4. Selected the required features:
   - Temperature
   - Relative Humidity
   - Surface Pressure
   - Wind Speed
5. Created the target column `Weather_Class`:
   - `Warm` if temperature is greater than or equal to 25°C
   - `Cool` if temperature is below 25°C
6. Checked for missing values and removed unnecessary columns.
7. Encoded the target variable.
8. Split the data into 80% training and 20% testing sets.
9. Standardized the features using `StandardScaler`.
10. Trained an SVM classifier with an RBF kernel.
11. Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results
Latest validated run:
- Accuracy: 0.9704
- Precision: 0.9556
- Recall: 0.9556
- F1-Score: 0.9556
- Confusion Matrix:
  - `[[88, 2], [2, 43]]`

## Conclusion
This experiment showed that Open-Meteo data can be used to classify weather conditions into Cool and Warm categories with an SVM classifier. Temperature was the most important feature because it directly defined the target label, while humidity, pressure, and wind speed helped shape the decision boundary. Feature scaling was critical for SVM because the algorithm is sensitive to differences in feature magnitude; without scaling, larger valued variables can dominate the model. One advantage of SVM is that it performs well on small to medium-sized datasets with clear class separation. One limitation is that it can become less interpretable and less efficient on large datasets or highly overlapping classes.

## Files Included
- `Assignment-6.py`
- `README.md`
# Assignment-6
