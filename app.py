import os
import io
import base64
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, session, flash, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import mysql.connector
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"

# ---------------- Database ----------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="dengue"
    )

# ---------------- Prediction Functions ----------------
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data['Week'] = pd.to_datetime(data['Week'])
    data.sort_values(by='Week', inplace=True)
    data.set_index('Week', inplace=True)
    return data

def modwt_decompose(series, wavelet='db4', level=4):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    detail_coeffs = coeffs[:-1]
    approx_coeff = coeffs[-1]
    return detail_coeffs, approx_coeff

def normalize_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler

def prepare_last_inputs(coeffs, lags=4):
    last_inputs = []
    scalers = []
    for series in coeffs:
        scaled, scaler = normalize_series(series)
        scalers.append(scaler)
        last_inputs.append(scaled[-lags:])
    return last_inputs, scalers

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

def forecast_lstm(models, last_values, scalers, steps=10, lags=4):
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

def create_plot(original, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-len(forecast):], label='Actual Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title('Dengue Cases Forecast')
    plt.xlabel('Weeks')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the image to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, phone, password) VALUES (%s, %s, %s, %s)",
                       (name, email, phone, hashed_password))
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
            flash("Login successful!", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash("Please login to access this page.", "warning")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            forecast_steps = int(request.form['weeks'])
            
            # Load and preprocess data
            san_juan_data = load_and_preprocess('Sanjuan_data_weekly.csv')
            san_juan_data['Cases'] = san_juan_data['Cases'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            # Decompose
            detail_coeffs, approx_coeff = modwt_decompose(san_juan_data['Cases'].values)
            all_coeffs = detail_coeffs + [approx_coeff]
            
            # Prepare inputs
            lags = 4
            last_values, scalers = prepare_last_inputs(all_coeffs, lags=lags)
            
            # Load models (assuming they're in the same directory)
            models = [load_model(f"lstm_model_{i}.h5") for i in range(len(all_coeffs))]
            
            # Forecast
            forecast = forecast_lstm(models, last_values, scalers, steps=forecast_steps, lags=lags)
            
            # Create plot
            plot_url = create_plot(san_juan_data['Cases'].values, forecast)
            
            # Determine trend
            trend = "increase" if forecast[-1] > forecast[0] else "decrease"
            
            # Calculate metrics
            original = san_juan_data['Cases'].values[-len(forecast):]
            mae = mean_absolute_error(original, forecast)
            rmse = np.sqrt(mean_squared_error(original, forecast))
            smape_value = smape(original, forecast)
            mase_value = mase(original, forecast)
            
            return render_template('predict.html', 
                                 plot_url=plot_url, 
                                 forecast=forecast.tolist(),
                                 trend=trend,
                                 mae=round(mae, 2),
                                 rmse=round(rmse, 2),
                                 smape=round(smape_value, 2),
                                 mase=round(mase_value, 2))
            
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)