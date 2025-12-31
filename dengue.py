import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Seaborn style for plots
sns.set(style="whitegrid")

# Load datasets
# Update the paths below with the actual file paths after downloading
san_juan_data = pd.read_csv('Ahmedabad_data_weekly.csv')
iquitos_data = pd.read_csv('Iquitos_data_weekly.csv')
ahmedabad_data = pd.read_csv('Sanjuan_data_weekly.csv')

# Display the first few rows to understand the structure
print("San Juan Data:\n", san_juan_data.head())
print("\nIquitos Data:\n", iquitos_data.head())
print("\nAhmedabad Data:\n", ahmedabad_data.head())

# Convert the 'week_start_date' to datetime
san_juan_data['week_start_date'] = pd.to_datetime(san_juan_data['week_start_date'])
iquitos_data['week_start_date'] = pd.to_datetime(iquitos_data['week_start_date'])
ahmedabad_data['week_start_date'] = pd.to_datetime(ahmedabad_data['week_start_date'])

# Plotting Dengue Cases and Rainfall over time
plt.figure(figsize=(18, 6))

# San Juan
plt.subplot(1, 3, 1)
plt.plot(san_juan_data['week_start_date'], san_juan_data['total_cases'], color='red', label='Dengue Cases')
plt.plot(san_juan_data['week_start_date'], san_juan_data['rainfall_mm'], color='blue', label='Rainfall (mm)')
plt.title('San Juan - Dengue Cases vs Rainfall')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()

# Iquitos
plt.subplot(1, 3, 2)
plt.plot(iquitos_data['week_start_date'], iquitos_data['total_cases'], color='red', label='Dengue Cases')
plt.plot(iquitos_data['week_start_date'], iquitos_data['rainfall_mm'], color='blue', label='Rainfall (mm)')
plt.title('Iquitos - Dengue Cases vs Rainfall')
plt.xlabel('Year')
plt.legend()

# Ahmedabad
plt.subplot(1, 3, 3)
plt.plot(ahmedabad_data['week_start_date'], ahmedabad_data['total_cases'], color='red', label='Dengue Cases')
plt.plot(ahmedabad_data['week_start_date'], ahmedabad_data['rainfall_mm'], color='blue', label='Rainfall (mm)')
plt.title('Ahmedabad - Dengue Cases vs Rainfall')
plt.xlabel('Year')
plt.legend()

plt.tight_layout()
plt.show()
'''


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------
# Step 1: Data Preprocessing
# ---------------------------
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data['Week'] = pd.to_datetime(data['Week'])
    data.sort_values(by='Week', inplace=True)
    data.set_index('Week', inplace=True)
    return data

# Load datasets
ahmedabad_data = load_and_preprocess('Ahmedabad_data_weekly.csv')
iquitos_data = load_and_preprocess('Iquitos_data_weekly.csv')
san_juan_data = load_and_preprocess('Sanjuan_data_weekly.csv')

# ---------------------------
# Step 2: MODWT Decomposition
# ---------------------------
def modwt_decompose(series, wavelet='db1', level=3):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    detail_coeffs = coeffs[:-1]
    approx_coeff = coeffs[-1]
    return detail_coeffs, approx_coeff

# Decompose San Juan data
detail_coeffs, approx_coeff = modwt_decompose(san_juan_data['Cases'].values)

# ---------------------------
# Step 3: Train Random Forest Models
# ---------------------------
def create_supervised_data(series, lags=4):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_random_forest(coeffs, lags=4):
    models = []
    last_values = []

    for series in coeffs:
        X, y = create_supervised_data(series, lags)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models.append(model)
        last_values.append(series[-lags:])  # Store last values for forecasting

    return models, last_values

# Train Random Forest on wavelet detail coefficients
models, last_values = train_random_forest(detail_coeffs)

# ---------------------------
# Step 4: Forecast Using Random Forest
# ---------------------------
def forecast_random_forest(models, last_values, steps=10, lags=4):
    forecasts = []
    for model, last in zip(models, last_values):
        current_input = last.copy()
        forecast = []

        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            forecast.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred

        forecasts.append(forecast)

    ensemble_result = np.sum(forecasts, axis=0)
    return ensemble_result

# Forecast next 10 weeks
forecast = forecast_random_forest(models, last_values, steps=52)

# ---------------------------
# Step 5: Visualization and Evaluation
# ---------------------------
def plot_forecast(original, forecast, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-len(forecast):], label='Actual Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def smape(original, forecast):
    """ Symmetric Mean Absolute Percentage Error """
    return 100 * np.mean(2 * np.abs(forecast - original) / (np.abs(original) + np.abs(forecast)))

def mase(original, forecast, seasonality=1):
    """ Mean Absolute Scaled Error """
    n = len(original)
    d = np.abs(np.diff(original, n=seasonality)).sum() / (n - seasonality)
    errors = np.abs(original - forecast)
    return errors.mean() / d

def evaluate_model(original, forecast):
    original, forecast = np.array(original), np.array(forecast)
    
    print('MAE:', mean_absolute_error(original[-len(forecast):], forecast))
    print('RMSE:', np.sqrt(mean_squared_error(original[-len(forecast):], forecast)))
    print('SMAPE:', smape(original[-len(forecast):], forecast))
    print('MASE:', mase(original[-len(forecast):], forecast))


    
   

# Visualize and Evaluate
plot_forecast(san_juan_data['Cases'].values, forecast)
evaluate_model(san_juan_data['Cases'].values, forecast)










