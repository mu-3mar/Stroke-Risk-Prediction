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
                    public/ (separate server locally, CDN on Vercel)
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
├── public/
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

Install the production and notebook dependencies for training, analysis, and
the API:

```bash
pip install -r requirements-dev.txt
```

[`requirements.txt`](requirements.txt) contains only the dependencies needed by
the deployed inference API. [`requirements-dev.txt`](requirements-dev.txt)
includes that production set plus Jupyter and the notebook visualization
packages (`matplotlib` and `seaborn`).

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
python3 -m http.server 5500 --directory public
```

Open <http://127.0.0.1:5500/> in a browser. During local development, the
frontend connects to the API at `http://127.0.0.1:8000`; if the API is
stopped, the frontend remains open and shows the API as offline. On Vercel,
the same frontend uses same-origin requests to the deployed API. The API also
exposes `/health` and `POST /predict`.

The frontend in [`public/`](public) uses the same 16 inputs as the trained
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

## Vercel deployment

This repository uses Vercel's native Python runtime. Vercel discovers the
FastAPI instance exported as `app` from [`src/main.py`](src/main.py), installs
the runtime dependencies from [`requirements.txt`](requirements.txt), and
serves the files in [`public/`](public/) from the CDN. [`vercel.json`](vercel.json)
explicitly includes both serialized model artifacts in the Python function
bundle.

Connect the repository to Vercel with the repository root as the project root.
No build command or output directory is required. After deployment, open the
deployment URL and verify `/health`, `/docs`, and the browser form. The
prediction API remains `POST /predict`; frontend requests are relative to the
deployment URL.

Vercel functions are stateless and may cold-start while importing the model.
The serialized artifacts are approximately 660 KB in total, but the Python
runtime and scientific dependencies still contribute to bundle size and cold
start time. The deployment does not require the training dataset: training
and notebook processing are not imported by the API.

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
