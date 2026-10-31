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
├── requirements-docker.txt
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

Install all packages for training, analysis, and the API:

```bash
pip install -r requirements.txt
```

For the API Docker image, install only the runtime dependencies:

```bash
pip install -r requirements-docker.txt
```

The Docker requirements exclude the notebook and visualization packages
(`jupyter`, `matplotlib`, and `seaborn`). They still include
`scikit-learn`, because it is required to load and run the trained model
artifacts.

## Dockerization

The FastAPI backend has been containerized and the image is available on
[Docker Hub](https://hub.docker.com/r/mu3mar/stroke-risk-api):

```text
mu3mar/stroke-risk-api:latest
```

Pull and run the image:

```bash
docker pull mu3mar/stroke-risk-api:latest
docker run -d \
  --name stroke-risk-api \
  -p 8000:8000 \
  mu3mar/stroke-risk-api:latest
```

After starting the container, check the API health:

```bash
curl http://localhost:8000/health
```

The API documentation is available at
<http://localhost:8000/docs>. To stop and remove the container:

```bash
docker stop stroke-risk-api
docker rm stroke-risk-api
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
