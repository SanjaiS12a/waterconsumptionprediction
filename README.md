# 💧 Water Consumption Prediction & Demand Showdown

An interactive machine learning and deep learning framework to forecast daily water demand using Multi-Layer Perceptrons (MLP), Gated Recurrent Units (GRU), and Long Short-Term Memory (LSTM) networks. 

This repository features hyperparameter-tuned deep learning models, and a premium Streamlit-based comparison dashboard called **Water Demand Showdown**.

---

## 🚀 Key Features

* **Three-Model Showdown:** Compare the predictions of **MLP**, **GRU**, and **LSTM** models side-by-side.
* **Interactive Scenario Planning:** Simulate weather patterns, population changes, reservoir levels, holiday spikes, and pump outages to see how demand changes.
* **Robust Safety Checks:** Built-in validation rules (e.g., "Ghost Town Sanity Rule" that forces logical zero demand if population or household counts drop to zero).
* **Consensus & Spread Analytics:** Computes the average forecast, model consensus band, and the variance (spread) between models.

---

## 📁 Repository Structure

```
├── .gitignore                    # Git ignore rules (excludes venv, cache, tuning logs)
├── README.md                     # Project documentation (this file)
├── app.py                        # Streamlit dashboard interface & interactive forecasting
├── water_consumption_dataset.csv # The generated 6-year water consumption dataset
│
├── MLP.ipynb                     # MLP model training and evaluation
├── MLPHYP.ipynb                  # MLP hyperparameter tuning
├── GRU..ipynb                    # GRU model training and evaluation
├── gru hyp.ipynb                 # GRU hyperparameter tuning
├── modellstm.ipynb               # LSTM model training and evaluation
├── lstmhyp.ipynb                 # LSTM hyperparameter tuning
│
├── mlp_model.keras               # Trained Keras model for MLP
├── gru_model.keras               # Trained Keras model for GRU
├── lstm_model.keras              # Trained Keras model for LSTM
├── feature_scaler.pkl            # Pre-saved feature scaler
└── target_scaler.pkl             # Pre-saved target scaler
```

---

## 📊 Dataset & Feature Engineering

The dataset spans 6 years of daily data (2020–2025) and includes the following features:

| Feature Category | Features Included |
| :--- | :--- |
| **Temporal** | Day of Week, Month, Year, Day of Year, Is Weekend, Holiday Flag, sin/cos cyclical encodings |
| **Meteorological** | Temperature (°C), Rainfall (mm), Humidity (%), Evaporation Rate (mm) |
| **Operational & Socio-economic**| Population, Household Count, Urbanization Index, Pump Status, Reservoir Level (%) |
| **Historical Lags** | Past demands (`lag_1` to `lag_365`), moving averages (`ma_7`, `ma_30`), rolling std. dev. |

---

## 🤖 Models & Sequence Windows

Each model requires a different historical window to predict the target, `Total_Daily_Usage_ML` (Total Daily Usage in Megaliters):

1. **MLP (Multi-Layer Perceptron):**
   * *Lookback Window:* 1 day
   * *Features:* 42 features (weather, lags, temporal, operational)
2. **GRU (Gated Recurrent Unit):**
   * *Lookback Window:* 14 days (sequence model)
   * *Features:* 42 features
3. **LSTM (Long Short-Term Memory):**
   * *Lookback Window:* 60 days (sequence model)
   * *Features:* 39 features

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Install the required libraries:
```bash
pip install streamlit pandas numpy tensorflow keras scikit-learn
```

### 3. Run the Streamlit Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Tour ("Water Demand Showdown")

* **Date Selector:** Pick any date from the dataset range up to the year 2035.
* **Scenario Inputs:** Tune parameters like Temperature, Rainfall, Population, and Pump Status in real time.
* **Live Prediction Analytics:** Run the models to calculate:
  * **Consensus Demand (ML):** The average predicted demand across all active models.
  * **Spread (ML):** The difference between the highest and lowest predictions.
  * **Consensus Band:** Classifies the consensus value into `Low` (< 80 ML), `Medium` (80-110 ML), or `High` (> 110 ML) demand levels.
* **Visualization:** Renders an interactive bar chart of model outputs alongside a clear summary panel indicating the closest-to-consensus model.
