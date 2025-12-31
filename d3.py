import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

'''
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
# Step 3: Data Preparation for LSTM
# ---------------------------
def create_supervised_data(series, lags=4):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Normalize the data (scaling between 0 and 1)
scalers = []

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    scalers.append(scaler)
    return data_scaled

# Prepare data for LSTM
def prepare_lstm_data(coeffs, lags=4):
    X_data, y_data, last_values = [], [], []
    
    for series in coeffs:
        series_scaled = normalize_data(series)
        X, y = create_supervised_data(series_scaled, lags)
        X_data.append(X)
        y_data.append(y)
        last_values.append(series_scaled[-lags:])
        
    return X_data, y_data, last_values

X_data, y_data, last_values = prepare_lstm_data(detail_coeffs)

# ---------------------------
# Step 4: Build LSTM Model
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False)) 
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Adjust optimizer
    return model

# Build LSTM models for each coefficient series
models = []
for i, (X, y) in enumerate(zip(X_data, y_data)):
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)
    models.append(model)
    
    # Save the model
    model.save(f"lstm_model_{i}.h5")  # or use .keras for newer format

# ---------------------------
# Step 5: Forecast Using LSTM
# ---------------------------
def forecast_lstm(models, last_values, steps=10, lags=4):
    forecasts = []
    for model, last, scaler in zip(models, last_values, scalers):
        current_input = last.reshape(1, lags, 1)
        forecast = []

        for _ in range(steps):
            pred = model.predict(current_input)[0, 0]
            forecast.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[0, -1, 0] = pred

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecasts.append(forecast)

    ensemble_result = np.sum(forecasts, axis=0)
    return ensemble_result

# Forecast next weeks
forecast = forecast_lstm(models, last_values, steps=int(input("Enter number of weeks: ")))

# ---------------------------
# Step 6: Trend Direction Classification
# ---------------------------
forecast_changes = np.diff(forecast)
trend_direction = np.where(forecast_changes > 0, 1, 0)

# ---------------------------
# Step 7: Evaluation
# ---------------------------
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    # Avoid division by zero by checking if both true and predicted values are non-zero.
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0  # Handle division by zero case
    return 100 * np.mean(diff)

def mase(y_true, y_pred, seasonality=52):
    """Mean Absolute Scaled Error"""
    n = len(y_true)
    
    # Adjust seasonality if there's not enough data
    if n <= seasonality:
        seasonality = n // 2  # Use half the dataset length or adjust as needed
    
    # Calculate the scaling factor `d`
    d = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    
    # Handle case where `d` is zero (which would make MASE invalid)
    if d == 0:
        return np.nan  # Or return a very large value, e.g., return float('inf')
    
    # Calculate the errors and return the scaled mean absolute error
    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / d


def evaluate_model(original, forecast, seasonality=52):
    original = np.array(original[-len(forecast):])  # Ensure alignment with forecast length
    forecast = np.array(forecast)
    
    mae = mean_absolute_error(original, forecast)
    rmse = np.sqrt(mean_squared_error(original, forecast))
    smape_value = smape(original, forecast)
    mase_value = mase(original, forecast, seasonality)
    
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'SMAPE: {smape_value:.4f}')
    print(f'MASE: {mase_value:.4f}')

evaluate_model(san_juan_data['Cases'].values, forecast)

# ---------------------------
# Step 8: Visualization
# ---------------------------
def plot_forecast(original, forecast, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-len(forecast):], label='Actual Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_forecast(san_juan_data['Cases'].values, forecast)

def display_trend(forecast):
    if forecast[-1] > forecast[0]:
        print("Dengue will **INCREASE**.")
    else:
        print("Dengue will **DECREASE**.")

# Call the function after plotting
display_trend(forecast)


'''











'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
def modwt_decompose(series, wavelet='db4', level=2):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    detail_coeffs = coeffs[:-1]
    approx_coeff = coeffs[-1]
    return detail_coeffs, approx_coeff

detail_coeffs, approx_coeff = modwt_decompose(san_juan_data['Cases'].values)
all_coeffs = detail_coeffs + [approx_coeff]  # Use all wavelet components

# ---------------------------
# Step 3: Data Preparation for LSTM
# ---------------------------
def create_supervised_data(series, lags=4):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

scalers = []

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    scalers.append(scaler)
    return data_scaled

def prepare_lstm_data(coeffs, lags=4):
    X_data, y_data, last_values = [], [], []

    for series in coeffs:
        series_scaled = normalize_data(series)
        X, y = create_supervised_data(series_scaled, lags)
        X_data.append(X)
        y_data.append(y)
        last_values.append(series_scaled[-lags:])
    
    return X_data, y_data, last_values

X_data, y_data, last_values = prepare_lstm_data(all_coeffs)

# ---------------------------
# Step 4: Build LSTM Model
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    return model

models = []
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

for i, (X, y) in enumerate(zip(X_data, y_data)):
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Ensure 3D input for LSTM
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=1, callbacks=[early_stop])
    models.append(model)
    model.save(f"lstm_model_{i}.h5")

# ---------------------------
# Step 5: Forecast Using LSTM
# ---------------------------
def forecast_lstm(models, last_values, steps=10, lags=4):
    forecasts = []
    for model, last, scaler in zip(models, last_values, scalers):
        current_input = last.reshape(1, lags, 1)
        forecast = []

        for _ in range(steps):
            pred = model.predict(current_input, verbose=0)[0, 0]
            forecast.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[0, -1, 0] = pred

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecasts.append(forecast)

    ensemble_result = np.sum(forecasts, axis=0)
    ensemble_result = np.clip(ensemble_result, 0, None)  # Remove negative predictions
    return ensemble_result

# Forecast next weeks
forecast_steps = int(input("Enter number of weeks to forecast: "))
forecast = forecast_lstm(models, last_values, steps=forecast_steps)

# ---------------------------
# Step 6: Trend Direction Classification
# ---------------------------
forecast_changes = np.diff(forecast)
trend_direction = np.where(forecast_changes > 0, 1, 0)

# ---------------------------
# Step 7: Evaluation
# ---------------------------
def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true)
    smape_val = np.where(denominator == 0, 0, diff / denominator)
    return 100 * np.mean(smape_val)

def mase(y_true, y_pred, seasonality=52):
    n = len(y_true)
    if n <= seasonality:
        seasonality = max(1, n // 2)
    d = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    if d == 0:
        return np.nan
    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / d

def evaluate_model(original, forecast, seasonality=52):
    original = np.array(original[-len(forecast):])
    forecast = np.array(forecast)
    
    mae = mean_absolute_error(original, forecast)
    rmse = np.sqrt(mean_squared_error(original, forecast))
    smape_value = smape(original, forecast)
    mase_value = mase(original, forecast, seasonality)
    
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'SMAPE: {smape_value:.4f}')
    print(f'MASE: {mase_value:.4f}')

evaluate_model(san_juan_data['Cases'].values, forecast)

# ---------------------------
# Step 8: Visualization
# ---------------------------
def plot_forecast(original, forecast, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-len(forecast):], label='Actual Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_forecast(san_juan_data['Cases'].values, forecast)

def display_trend(forecast):
    if forecast[-1] > forecast[0]:
        print("Dengue will **INCREASE**.")
    else:
        print("Dengue will **DECREASE**.")

display_trend(forecast)

'''







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

# Optional: Smooth data to reduce noise
san_juan_data['Cases'] = san_juan_data['Cases'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# ---------------------------
# Step 2: MODWT Decomposition
# ---------------------------
def modwt_decompose(series, wavelet='db4', level=4):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    detail_coeffs = coeffs[:-1]
    approx_coeff = coeffs[-1]
    return detail_coeffs, approx_coeff

detail_coeffs, approx_coeff = modwt_decompose(san_juan_data['Cases'].values)
all_coeffs = detail_coeffs + [approx_coeff]

# ---------------------------
# Step 3: Data Preparation
# ---------------------------
def create_supervised_data(series, lags=4):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

scalers = []

def normalize_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    scalers.append(scaler)
    return scaled

def prepare_lstm_data(coeffs, lags=4):
    X_data, y_data, last_values = [], [], []
    for series in coeffs:
        scaled = normalize_data(series)
        X, y = create_supervised_data(scaled, lags)
        X_data.append(X)
        y_data.append(y)
        last_values.append(scaled[-lags:])
    return X_data, y_data, last_values

X_data, y_data, last_values = prepare_lstm_data(all_coeffs)

# ---------------------------
# Step 4: Build LSTM Model
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

models = []
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for i, (X, y) in enumerate(zip(X_data, y_data)):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stop])
    models.append(model)
    model.save(f"lstm_model_{i}.h5")

# ---------------------------
# Step 5: Forecasting
# ---------------------------
def forecast_lstm(models, last_values, steps=10, lags=4):
    forecasts = []
    for model, last, scaler in zip(models, last_values, scalers):
        current_input = last.reshape(1, lags, 1)
        forecast = []
        for _ in range(steps):
            pred = model.predict(current_input, verbose=0)[0, 0]
            forecast.append(pred)
            current_input = np.roll(current_input, -1)
            current_input[0, -1, 0] = pred
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecasts.append(forecast)
    ensemble_result = np.sum(forecasts, axis=0)
    ensemble_result = np.clip(ensemble_result, 0, None)
    return ensemble_result

# Forecast next weeks
forecast_steps = int(input("Enter number of weeks to forecast: "))
forecast = forecast_lstm(models, last_values, steps=forecast_steps)

# ---------------------------
# Step 6: Trend Direction Classification
# ---------------------------
forecast_changes = np.diff(forecast)
trend_direction = np.where(forecast_changes > 0, 1, 0)

# ---------------------------
# Step 7: Evaluation Metrics
# ---------------------------
def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true)
    smape_val = np.where(denominator == 0, 0, diff / denominator)
    return 100 * np.mean(smape_val)

def mase(y_true, y_pred, seasonality=52):
    n = len(y_true)
    if n <= seasonality:
        seasonality = max(1, n // 2)
    d = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    if d == 0:
        return np.nan
    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / d

def evaluate_model(original, forecast, seasonality=52):
    original = np.array(original[-len(forecast):])
    forecast = np.array(forecast)
    mae = mean_absolute_error(original, forecast)
    rmse = np.sqrt(mean_squared_error(original, forecast))
    smape_value = smape(original, forecast)
    mase_value = mase(original, forecast, seasonality)
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'SMAPE: {smape_value:.4f}')
    print(f'MASE: {mase_value:.4f}')

evaluate_model(san_juan_data['Cases'].values, forecast)

# ---------------------------
# Step 8: Visualization
# ---------------------------
def plot_forecast(original, forecast, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-len(forecast):], label='Actual Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_forecast(san_juan_data['Cases'].values, forecast)

def display_trend(forecast):
    if forecast[-1] > forecast[0]:
        print("Dengue will **INCREASE**.")
    else:
        print("Dengue will **DECREASE**.")

display_trend(forecast)










