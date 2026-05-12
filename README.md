#  Dengue Outbreak Prediction System

Predicting dengue outbreaks is notoriously hard because the data is incredibly "noisy"—it jumps around due to weather, reporting delays, and seasonal shifts. I built this system to see if a hybrid deep learning approach could cut through that noise and predict cases 2-4 weeks in advance.

The goal wasn't just to make a model, but to create a full pipeline where you can actually see the predictions versus the real data.

###  How it Works 
Most models struggle with raw dengue data because it's too erratic. Here’s how I tackled it:

1.  **Cleaning the Signal (MODWT):** I used Wavelet Transforms to "decompose" the data. Think of it like taking a messy audio recording and separating the background noise from the actual melody. This helps the model focus on the real trend.
2.  **The Memory (LSTM):** Once the data was cleaned, I fed it into an LSTM (Long Short-Term Memory) network. Since dengue follows seasonal patterns, the LSTM is perfect for "remembering" what happened in previous months to predict the next few weeks.
3.  **The Dashboard:** I wrapped everything in a Flask app so you can visualize the predictions. It’s one thing to see numbers, but another to see the curve of an outbreak actually being forecasted.

---

###  Performance
During testing, the model was surprisingly good at catching the *direction* of an outbreak.
*   **Accuracy:** It caught the rising/falling trends about **90% of the time**.
*   **Error Rate:** On average, it was off by about **6.8 cases** (MAE). 
*   **The Win:** It’s particularly strong at identifying when a quiet period is about to turn into an outbreak phase.

---

###  Tech Stack
*   **Deep Learning:** TensorFlow / Keras (LSTM)
*   **Signal Processing:** PyWavelets (MODWT)
*   **Data:** Pandas, NumPy
*   **Web:** Flask & Matplotlib

---

###  Getting Started
1.  Clone this repo: `git clone https://github.com/vpriyathimma/dengue-outbreak-prediction.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the dashboard: `python app.py`
4.  Open your browser to `localhost:5000` and pick a region to see the forecast.




---
**Vishnupriya T**  
[vpriyathimma@gmail.com](mailto:vpriyathimma@gmail.com) | [@vpriyathimma](https://github.com/vpriyathimma)




