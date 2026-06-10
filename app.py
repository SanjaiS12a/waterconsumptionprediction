from __future__ import annotations

import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import keras
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "water_consumption_dataset.csv"
TARGET = "Total_Daily_Usage_ML"

MLP_FEATURES = [
    "Day_of_Week", "Month", "Year", "Day_of_Year", "Is_Weekend", "Holiday_Flag",
    "Temperature_C", "Rainfall_mm", "Humidity_pct", "Evaporation_Rate_mm",
    "Population", "Household_Count", "Urbanization_Index", "Pump_Status",
    "Reservoir_Level_pct", "month_sin", "month_cos", "day_sin", "day_cos",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_7", "lag_14", "lag_21",
    "lag_30", "lag_60", "lag_90", "lag_180", "lag_365", "ma_7", "ma_30",
    "std_7", "std_30", "ema_7", "ema_30", "diff_1", "diff_7", "pct_change_1",
    "pct_change_7",
]

LSTM_FEATURES = [
    "Day_of_Week", "Month", "Year", "Day_of_Year", "Is_Weekend", "Holiday_Flag",
    "Temperature_C", "Rainfall_mm", "Humidity_pct", "Evaporation_Rate_mm",
    "Population", "Household_Count", "Urbanization_Index", "Pump_Status",
    "Reservoir_Level_pct", "month_sin", "month_cos", "day_sin", "day_cos",
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_7", "lag_14", "lag_21", "lag_30",
    "lag_60", "lag_90", "lag_365", "ma_7", "ma_30", "std_7", "std_30",
    "rolling_max_7", "rolling_min_7", "diff_1", "diff_3", "diff_7",
]

MODEL_SPECS = {
    "MLP": {
        "path": BASE_DIR / "mlp_model.keras",
        "window": 1,
        "features": MLP_FEATURES,
        "kind": "mlp",
    },
    "GRU": {
        "path": BASE_DIR / "gru_model.keras",
        "window": 14,
        "features": MLP_FEATURES,
        "kind": "sequence",
    },
    "LSTM": {
        "path": BASE_DIR / "lstm_model.keras",
        "window": 60,
        "features": LSTM_FEATURES,
        "kind": "sequence",
    },
}

ALL_FEATURES = sorted({feature for spec in MODEL_SPECS.values() for feature in spec["features"]})


def sanitize_keras_config(node: object) -> None:
    if isinstance(node, dict):
        class_name = node.get("class_name")
        config = node.get("config")
        if isinstance(config, dict):
            config.pop("dtype", None)
            config.pop("quantization_config", None)

            if class_name == "InputLayer":
                batch_shape = config.pop("batch_shape", None)
                if batch_shape is not None and "batch_input_shape" not in config:
                    config["batch_input_shape"] = tuple(batch_shape)

            if class_name == "BatchNormalization":
                config.pop("renorm", None)
                config.pop("renorm_clipping", None)
                config.pop("renorm_momentum", None)

        for value in node.values():
            sanitize_keras_config(value)
    elif isinstance(node, list):
        for value in node:
            sanitize_keras_config(value)


def load_legacy_keras_model(model_path: Path) -> keras.Model:
    with ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))
        weights_blob = archive.read("model.weights.h5")

    sanitize_keras_config(config)
    model = keras.models.model_from_json(json.dumps(config))

    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as temp_weights:
        temp_weights.write(weights_blob)
        temp_weights_path = Path(temp_weights.name)

    try:
        model.load_weights(temp_weights_path)
    finally:
        temp_weights_path.unlink(missing_ok=True)

    return model


# --- CONFIG & MODERN SLATE THEME CSS ---
st.set_page_config(
    page_title="Water Demand Showdown",
    page_icon="💧",
    layout="wide",
)

st.markdown(
    """
<style>
/* App Deep Slate Architecture */
.stApp { 
    background: #0f172a !important; 
    color: #f8fafc !important; 
}
.stApp, .stApp * { 
    color: #f8fafc !important; 
}

/* Premium Hero Header Banner */
.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    color: #ffffff !important;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px -15px rgba(0, 0, 0, 0.5);
}
.hero h1 { 
    margin: 0; 
    font-size: 2.5rem; 
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { 
    margin: 0.5rem 0 0; 
    opacity: 0.85; 
    font-size: 1.05rem; 
    color: #cbd5e1 !important;
}

/* Modern Card Containers */
.soft-card {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
}
.mini-label { 
    font-size: 0.85rem; 
    font-weight: 700;
    color: #38bdf8 !important;
    text-transform: uppercase; 
    letter-spacing: 0.12em; 
    margin-bottom: 1.25rem; 
}

/* Inputs, Selectboxes & Interactive Components */
.stApp input, .stApp textarea, .stApp select, .stApp .stDateInput input, .stApp .stNumberInput input {
    background-color: #0f172a !important; 
    color: #f8fafc !important; 
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
}
.stApp input:focus, .stApp select:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Action Button styling Overhaul */
.stApp .stButton button {
    background: linear-gradient(180deg, #38bdf8 0%, #0284c7 100%) !important; 
    color: #ffffff !important; 
    border: none !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}
.stApp .stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
}

/* Base Metrics Formatting override */
.stMetric {
    background: #0f172a !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.stMetric, .stCaption, .stText { color: #f8fafc !important; }
.stApp .element-container { background: transparent !important; }
.stApp a { color: #38bdf8 !important; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    data["Day_of_Week"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    data["Day_of_Year"] = data["Date"].dt.dayofyear
    data["Is_Weekend"] = (data["Day_of_Week"] >= 5).astype(int)

    data["month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    data["day_sin"] = np.sin(2 * np.pi * data["Day_of_Year"] / 365)
    data["day_cos"] = np.cos(2 * np.pi * data["Day_of_Year"] / 365)

    history = data[TARGET].shift(1)
    for lag in [1, 2, 3, 4, 5, 7, 14, 21, 30, 60, 90, 180, 365]:
        data[f"lag_{lag}"] = data[TARGET].shift(lag)

    data["ma_7"] = history.rolling(7).mean()
    data["ma_30"] = history.rolling(30).mean()
    data["std_7"] = history.rolling(7).std()
    data["std_30"] = history.rolling(30).std()
    data["ema_7"] = history.ewm(span=7, adjust=False).mean()
    data["ema_30"] = history.ewm(span=30, adjust=False).mean()
    data["rolling_max_7"] = history.rolling(7).max()
    data["rolling_min_7"] = history.rolling(7).min()
    data["diff_1"] = history.diff(1)
    data["diff_3"] = history.diff(3)
    data["diff_7"] = history.diff(7)
    data["pct_change_1"] = history.pct_change(1)
    data["pct_change_7"] = history.pct_change(7)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


@st.cache_data(show_spinner=False)
def prepare_common_frame() -> pd.DataFrame:
    raw = load_data()
    feature_frame = build_feature_frame(raw)
    valid = feature_frame.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    return valid


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for name, spec in MODEL_SPECS.items():
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing model file: {spec['path']}")
        models[name] = load_legacy_keras_model(spec["path"])
    return models


@st.cache_resource(show_spinner=False)
def build_scalers() -> dict[str, dict[str, MinMaxScaler]]:
    frame = prepare_common_frame()
    split_idx = int(len(frame) * 0.8)
    train_frame = frame.iloc[:split_idx]

    scalers: dict[str, dict[str, MinMaxScaler]] = {}
    for name, spec in MODEL_SPECS.items():
        feature_scaler = MinMaxScaler().fit(train_frame[spec["features"]])
        target_scaler = MinMaxScaler().fit(train_frame[[TARGET]])
        scalers[name] = {
            "feature": feature_scaler,
            "target": target_scaler,
        }
    return scalers


def demand_band(value: float) -> str:
    if value < 80:
        return "Low"
    if value < 110:
        return "Medium"
    return "High"


def load_optional_outputs() -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame | None]:
    metrics_path = BASE_DIR / "outputs" / "model_metrics.json"
    predictions_path = BASE_DIR / "outputs" / "test_predictions.csv"
    forecast_path = BASE_DIR / "outputs" / "30day_forecast.csv"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
    forecast = pd.read_csv(forecast_path) if forecast_path.exists() else None

    return predictions, metrics, forecast


def predict_model(model_name: str, frame: pd.DataFrame, forecast_date: pd.Timestamp) -> float:
    spec = MODEL_SPECS[model_name]
    scalers = build_scalers()[model_name]
    model = load_models()[model_name]

    row_positions = frame.index[frame["Date"] == forecast_date].tolist()
    if not row_positions:
        raise ValueError(f"Date {forecast_date.date()} is not available in the dataset.")

    row_idx = row_positions[0]

    if spec["kind"] == "mlp":
        input_frame = frame.loc[[row_idx], spec["features"]]
        scaled_input = scalers["feature"].transform(input_frame)
        prediction = model.predict(scaled_input, verbose=0)
    else:
        window = spec["window"]
        if row_idx < window:
            raise ValueError(f"Not enough history for {model_name}. Need at least {window} prior rows.")
        
        input_frame = frame.iloc[row_idx - window : row_idx].copy()
        input_frame = pd.concat([input_frame, frame.loc[[row_idx]]], ignore_index=True).tail(window)
        
        scaled_input = scalers["feature"].transform(input_frame[spec["features"]])
        scaled_input = scaled_input.reshape(1, window, len(spec["features"]))
        prediction = model.predict(scaled_input, verbose=0)

    return float(scalers["target"].inverse_transform(np.asarray(prediction).reshape(-1, 1))[0, 0])


def prepare_prediction_frame(frame: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    working_frame = frame.copy()
    if (working_frame["Date"] == forecast_date).any():
        return build_feature_frame(working_frame)

    future_row = working_frame.iloc[[-1]].copy()
    future_row["Date"] = forecast_date
    future_row[TARGET] = np.nan
    extended_frame = pd.concat([working_frame, future_row], ignore_index=True)
    return build_feature_frame(extended_frame)


def main() -> None:
    frame = prepare_common_frame()
    optional_predictions, optional_metrics, optional_forecast = load_optional_outputs()

    st.markdown(
        """
        <div class="hero">
            <h1>Water Demand Showdown</h1>
            <p>Compare MLP, GRU, and LSTM forecasts for a selected date using the trained models in this workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- FULL WIDTH CARD BLOCK ---
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Forecast configuration setup</div>', unsafe_allow_html=True)

    min_date = frame["Date"].min().date()
    max_date = pd.Timestamp("2035-12-31").date()
    default_date = max(min_date, pd.Timestamp("2026-01-01").date())

    selected_date = st.date_input("Required date", value=default_date, min_value=min_date, max_value=max_date)
    
    date_ts = pd.Timestamp(selected_date)
    base_prediction_frame = prepare_prediction_frame(frame, date_ts)
    row = base_prediction_frame.loc[base_prediction_frame["Date"] == date_ts].iloc[0]

    with st.form("forecast_form"):
        mode = st.radio("Model mode", ["Showdown", "Single model"], horizontal=True)

        chosen_model = None
        if mode == "Single model":
            chosen_model = st.selectbox("Pick a model", list(MODEL_SPECS.keys()))

        st.markdown("<p style='font-weight: 600; margin-top: 1rem; color: #cbd5e1;'>Scenario Matrix Inputs</p>", unsafe_allow_html=True)
        
        # Grid System inside Form to elegantly present data metrics across full width
        f_col1, f_col2 = st.columns(2, gap="medium")
        with f_col1:
            temperature = st.number_input("Temperature (C)", value=float(row['Temperature_C']), format="%.1f")
            rainfall = st.number_input("Rainfall (mm)", value=float(row['Rainfall_mm']), format="%.1f")
            humidity = st.number_input("Humidity (%)", value=float(row['Humidity_pct']), format="%.1f")
            population = st.number_input("Population", value=int(row["Population"]), min_value=0, step=1)
        with f_col2:
            household_count = st.number_input("Household count", value=int(row["Household_Count"]), min_value=0, step=1)
            pump_status = st.selectbox("Pump status", [0, 1], index=int(row["Pump_Status"]))
            reservoir_level = st.number_input("Reservoir level (%)", value=float(row['Reservoir_Level_pct']), format="%.2f")
            holiday_flag = st.selectbox("Holiday flag", [0, 1], index=int(row["Holiday_Flag"]))

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run showdown")

    st.markdown('</div>', unsafe_allow_html=True)

    if optional_metrics is not None:
        with st.expander("Saved evaluation files"):
            if isinstance(optional_metrics, dict):
                st.json(optional_metrics)
            if optional_predictions is not None:
                st.dataframe(optional_predictions.head(10), use_container_width=True)
            if optional_forecast is not None:
                st.dataframe(optional_forecast.head(10), use_container_width=True)

    if not submitted:
        st.info("Adjust inputs in the form above and click **Run showdown** to generate model predictions.")
        st.stop()

    st.divider()
    st.subheader("Prediction Analytics")

    model_names = list(MODEL_SPECS.keys()) if mode == "Showdown" else [chosen_model]

    # --- THE GHOST TOWN SANITY RULE ---
    if population == 0 or household_count == 0:
        st.warning("⚠️ Population or Household Count is zero. Intercepting model prediction and forcing logical zero demand.")
        
        results = [
            {
                "Model": name,
                "Predicted demand (ML)": 0.00,
                "Demand level": "Low",
            }
            for name in model_names
        ]
        
        metric_cols = st.columns(3)
        metric_cols[0].metric("Consensus", "0.00 ML")
        metric_cols[1].metric("Spread", "0.00 ML")
        metric_cols[2].metric("Consensus band", "Low")

        result_frame = pd.DataFrame(results)
        st.dataframe(result_frame, use_container_width=True, hide_index=True)
        st.stop()
    # ----------------------------------

    overrides = {
        "Temperature_C": temperature,
        "Rainfall_mm": rainfall,
        "Humidity_pct": humidity,
        "Evaporation_Rate_mm": float(row["Evaporation_Rate_mm"]),
        "Population": int(population),
        "Household_Count": int(household_count),
        "Urbanization_Index": float(row["Urbanization_Index"]),
        "Pump_Status": int(pump_status),
        "Reservoir_Level_pct": float(reservoir_level),
        "Holiday_Flag": int(holiday_flag),
        "Day_of_Week": int(date_ts.dayofweek),
        "Month": int(date_ts.month),
        "Year": int(date_ts.year),
        "Day_of_Year": int(date_ts.dayofyear),
        "Is_Weekend": int(date_ts.dayofweek >= 5),
        "month_sin": float(np.sin(2 * np.pi * date_ts.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * date_ts.month / 12)),
        "day_sin": float(np.sin(2 * np.pi * date_ts.dayofyear / 365)),
        "day_cos": float(np.cos(2 * np.pi * date_ts.dayofyear / 365)),
    }

    working_frame = prepare_prediction_frame(frame, date_ts)
    row_idx = working_frame.index[working_frame["Date"] == date_ts].tolist()[0]
    for column, value in overrides.items():
        working_frame.at[row_idx, column] = value

    results = []

    with st.spinner("Running the models..."):
        for model_name in model_names:
            try:
                prediction_value = predict_model(model_name, working_frame, date_ts)
                results.append(
                    {
                        "Model": model_name,
                        "Predicted demand (ML)": round(prediction_value, 2),
                        "Demand level": demand_band(prediction_value),
                    }
                )
            except Exception as exc:
                st.error(f"{model_name} failed: {exc}")

    if results:
        result_frame = pd.DataFrame(results)
        consensus_value = float(result_frame["Predicted demand (ML)"].mean())
        spread_value = float(result_frame["Predicted demand (ML)"].max() - result_frame["Predicted demand (ML)"].min())

        metric_cols = st.columns(3)
        metric_cols[0].metric("Consensus", f"{consensus_value:.2f} ML")
        metric_cols[1].metric("Spread", f"{spread_value:.2f} ML")
        metric_cols[2].metric("Consensus band", demand_band(consensus_value))

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.dataframe(result_frame, use_container_width=True, hide_index=True)
        st.bar_chart(result_frame.set_index("Model")["Predicted demand (ML)"])

        if mode == "Showdown":
            winner = result_frame.iloc[(result_frame["Predicted demand (ML)"].sub(consensus_value).abs()).argmin()]
            st.success(
                f"Closest-to-consensus model: {winner['Model']} at {winner['Predicted demand (ML)']:.2f} ML ({winner['Demand level']})"
            )
        else:
            single = result_frame.iloc[0]
            st.success(
                f"{single['Model']} predicts {single['Predicted demand (ML)']:.2f} ML, which lands in the {single['Demand level']} band."
            )


if __name__ == "__main__":
    main()