# Dengue Outbreak Prediction System

This project implements a time-series forecasting system to predict dengue outbreaks using historical epidemiological data. The objective is to support early warning and short-term outbreak monitoring by modeling temporal and seasonal trends in dengue incidence.

---

## Project Overview

- Built a hybrid time-series forecasting model combining wavelet decomposition and deep learning.
- Designed to predict dengue case trends 2–4 weeks ahead using weekly case data.
- Deployed as a Flask-based web application for visualization and evaluation.

---

## Methodology

- **Data Preprocessing**
  - Historical weekly dengue case data collected for multiple regions.
  - Noise reduction and trend extraction using MODWT (Maximal Overlap Discrete Wavelet Transform).

- **Model Architecture**
  - Hybrid MODWT + LSTM neural network.
  - LSTM captures temporal dependencies after wavelet-based decomposition.

- **Forecasting Horizon**
  - Short-term prediction of dengue cases for 2–4 weeks ahead.

---

## Model Performance

- Mean Absolute Error (MAE): **6.8 cases**
- Root Mean Squared Error (RMSE): **9.4 cases**
- Achieved over **90% trend accuracy** in identifying rising and declining outbreak phases during internal evaluation.

---

## Web Application

- Flask-based interface for:
  - Visualizing predicted vs actual dengue cases
  - Evaluating model performance
  - Exploring short-term outbreak trends



