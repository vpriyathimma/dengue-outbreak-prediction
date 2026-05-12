# Dengue Outbreak Prediction System

Dengue Outbreak Prediction System is a hybrid deep learning and signal-processing application designed to forecast dengue outbreak trends 2–4 weeks in advance by reducing noise in epidemiological time-series data.

Built to address the unpredictability of dengue case fluctuations caused by weather variability, delayed reporting, and seasonal changes, this project combines signal decomposition with deep learning to improve forecasting accuracy and outbreak preparedness.

By integrating MODWT-based signal cleaning, LSTM temporal forecasting, and a Flask-powered visualization dashboard, the system provides not only predictive intelligence but also interpretable outbreak trend analysis for real-world public health applications.

---

## Features

- Predicts dengue outbreak trends 2–4 weeks in advance  
- Hybrid MODWT + LSTM forecasting pipeline  
- Wavelet decomposition for noise reduction  
- Time-series forecasting using Long Short-Term Memory (LSTM)  
- Interactive Flask dashboard for region-based outbreak visualization  
- Forecast vs real-case comparison interface  
- Early outbreak trend identification  
- Data-driven epidemiological decision support  

---

## Tech Stack

### Deep Learning
- TensorFlow  
- Keras (LSTM)  

### Signal Processing
- PyWavelets (MODWT - Maximal Overlap Discrete Wavelet Transform)  

### Data Processing
- Pandas  
- NumPy  

### Web Framework
- Flask  

### Visualization
- Matplotlib  

### Development Environment
- Python  

---

## How It Works

1. Raw dengue data is collected and preprocessed  
2. MODWT decomposes noisy time-series data into cleaner trend components  
3. Cleaned signals are passed into an LSTM model  
4. LSTM captures seasonal memory and temporal dependencies  
5. Flask dashboard visualizes predictions against real outbreak curves  
6. Users can select regions and analyze forecast patterns interactively  

---

## Performance Metrics

### Model Performance:
- Trend Direction Accuracy: ~90%  
- Mean Absolute Error (MAE): ~6.8 cases  
- Strong early outbreak phase detection  
- Effective for identifying transition from low-risk to outbreak periods  

---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/vpriyathimma/dengue-outbreak-prediction.git
cd dengue-outbreak-prediction
```

### Development Requirements
- Python 3.9+  
- pip  
- Virtual environment (recommended)  

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Dashboard
```bash
python app.py
```

### Access the Application
```bash
http://localhost:5000
```

---

## Project Structure

```bash
dengue-outbreak-prediction/
│── app.py
│── requirements.txt
│── models/
│── data/
│── preprocessing/
│── templates/
│── static/
└── README.md
```

---

## Current Scope

### Core Focus:
- Dengue outbreak forecasting  
- Time-series trend prediction  
- MODWT noise reduction  
- LSTM seasonal forecasting  
- Interactive outbreak visualization  

### Planned Enhancements:
- Weather API integration  
- Multi-disease forecasting (malaria, chikungunya)  
- Geospatial outbreak heatmaps  
- Cloud deployment  
- Automated alert systems  

---

## Security & Privacy

- Local deployment support  
- No sensitive personal health data exposure  
- Region-level forecasting design  
- Safe analytics-focused architecture  
- Expandable for public health system deployment  

---

## Future Improvements

- Real-time health surveillance integration  
- GIS mapping dashboards  
- Transformer-based forecasting models  
- Multi-region comparative analytics  
- SMS/email outbreak alerts  
- Mobile dashboard support  

---

## Why This Project Matters

This project demonstrates the practical application of AI, epidemiology, and signal processing to solve real-world healthcare forecasting challenges.

It highlights expertise in:
- Time-Series Forecasting  
- Deep Learning (LSTM)  
- Signal Processing (Wavelets)  
- Public Health Analytics  
- Flask Development  
- Data Visualization  

---

## Key Innovation

Traditional outbreak prediction systems often fail because of noisy, unstable epidemiological data. This system’s integration of MODWT for signal decomposition significantly improves pattern clarity before forecasting, making predictions more reliable and interpretable.

---

## Author

**Vishnupriya T**

- GitHub: https://github.com/vpriyathimma  
- Email: vpriyathimma@gmail.com  
- LinkedIn: https://www.linkedin.com/in/vishnupriya-t-7a0b8925b/  

---

## License

This project is intended for educational, portfolio, healthcare analytics, and epidemiological forecasting purposes. You may modify and expand it for research or public health innovation.




