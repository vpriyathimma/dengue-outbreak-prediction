# Dengue Outbreak Prediction System

---

## Project Overview

This project focuses on short-term dengue outbreak forecasting using a hybrid
time-series modeling approach. The objective is to predict dengue case trends
2–4 weeks ahead using historical weekly incidence data and to provide an
interactive web interface for visualization and evaluation.

The work emphasizes understanding, implementing, and validating an end-to-end
machine learning pipeline for epidemiological forecasting.

---

## Methodology

### Data Preparation
- Utilized historical weekly dengue case data from multiple regions.
- Performed data cleaning, normalization, and time-series structuring.
- Applied Maximal Overlap Discrete Wavelet Transform (MODWT) to decompose the
  original signal into trend and detail components for noise reduction.

### Model Architecture
- Implemented a hybrid MODWT + LSTM framework.
- Wavelet-decomposed components are used as inputs to an LSTM neural network.
- The LSTM model captures temporal and seasonal dependencies in dengue incidence
  patterns.

### Forecasting Setup
- Designed for short-term forecasting with a prediction horizon of 2–4 weeks.
- Model outputs are reconstructed to obtain final dengue case predictions.

---

## Model Evaluation

Model performance was evaluated using standard regression metrics:

- Mean Absolute Error (MAE): **6.8 cases**
- Root Mean Squared Error (RMSE): **9.4 cases**

The model demonstrated strong short-term trend-following capability, achieving
over **90% trend accuracy** in identifying rising and declining outbreak phases
during internal evaluation.

---

## Web Application

A Flask-based web application was developed to support:

- Visualization of predicted vs. actual dengue case trends
- Interactive evaluation of model performance
- Easy experimentation with different regional datasets

The application provides an end-to-end demonstration of the forecasting
pipeline, from data input to result interpretation.

---

## Project Scope and Learning Outcomes

This project was implemented as a practical machine learning exercise to gain
hands-on experience with:

- Time-series preprocessing and decomposition techniques
- Deep learning models for sequential data
- Model evaluation and error analysis
- Deployment of ML models using Flask for real-world interaction

---

## Limitations and Future Work

- The current model relies solely on historical case counts.
- Incorporating climatic, environmental, and mobility features could improve
  predictive performance.
- Future extensions may include attention-based models, probabilistic
  forecasting, and cross-region generalization analysis.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, PyWavelets
- Flask
- Matplotlib



