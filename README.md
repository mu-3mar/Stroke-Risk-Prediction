# Stroke Risk Prediction

An end-to-end machine learning project that explores patient health indicators,
trains several classification models, and provides a FastAPI backend with a
browser-based frontend for estimating stroke risk.

> **Medical disclaimer:** This project is for educational purposes only. It is
> not a medical device or diagnosis. Always consult a qualified healthcare
> professional for medical advice.

## Project workflow

```text
data/raw/healthcare-dataset-stroke-data.csv
              │
              ▼
notebooks/01_stroke_risk_prediction.ipynb
              │
              ├── data exploration and preprocessing
              └── models/best_stroke_model.pkl
                  models/scaler.pkl
                              │
                              ▼
                     src/main.py ──► POST /predict
                              ▲
                              │
                    frontend/ (separate server)
```

The notebook contains the complete analysis, preprocessing, model comparison,
and artifact generation workflow. The FastAPI application loads the saved
artifacts and serves only the prediction API. The frontend is a separate,
independent static application that connects to the API when it is available.

## Project structure

```text
Stroke-Risk-Prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_stroke_risk_prediction.ipynb # EDA, preprocessing, training
├── src/
│   ├── __init__.py
│   └── main.py                         # FastAPI app and prediction endpoint
├── frontend/
│   ├── index.html                      # Browser prediction form
│   ├── app.js                          # 16-feature request and result handling
│   ├── styles.css                      # Frontend styling
│   └── logo-animated.html              # Frontend logo
├── models/
│   ├── best_stroke_model.pkl           # trained classifier
│   └── scaler.pkl                      # fitted Age scaler
└── data/
    ├── README.md                       # dataset source and placement
    ├── raw/                            # local dataset files (ignored)
    └── processed/                      # optional generated data (ignored)
```

The dataset and generated processed data are intentionally not committed.
Download instructions are in [`data/README.md`](data/README.md).

## Usage

Install the project's Python packages:

```bash
pip install -r requirements.txt
```

To reproduce the analysis and regenerate the model files:

1. Put `healthcare-dataset-stroke-data.csv` in `data/raw/`.
2. Open `notebooks/01_stroke_risk_prediction.ipynb` from the repository root.
3. Run all cells. The trained artifacts will be written to `models/`.

Run the backend API in one terminal:

```bash
uvicorn src.main:app --reload
```

Run the independent frontend in a second terminal:

```bash
python3 -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500/> in a browser. The frontend connects to the API
at `http://127.0.0.1:8000`; if the API is stopped, the frontend remains open
and shows the API as offline. The API also exposes `/health` and `POST
/predict`.

The frontend in [`frontend/`](frontend) uses the same 16 inputs as the trained
model and sends this request to `POST /predict`:

```json
{
  "features": {
    "Age": 30,
    "Chest Pain": 0,
    "Shortness of Breath": 0
  }
}
```

The response should contain the model output:

```json
{"prediction": 0, "probability": 0.12, "threshold": 0.5}
```

The prediction response contains the binary prediction and class-1
probability. The project no longer uses Streamlit.

## Features

- Exploratory data analysis and visualization.
- Preprocessing of demographic and health-related features.
- Comparison of Logistic Regression, Random Forest, SVM, Decision Tree, and
  Gradient Boosting classifiers.
- Risk probability and binary risk prediction in a browser UI.

## Dataset

The project uses the public
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
from Kaggle. See [`data/README.md`](data/README.md) for the expected filename
and target definition.
