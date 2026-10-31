# Stroke Risk Prediction

An end-to-end machine-learning project for estimating stroke-risk likelihood
from age and health/symptom indicators. The project includes exploratory data
analysis, preprocessing, model comparison, a serialized scikit-learn model, a
FastAPI inference service, a responsive browser interface, Docker support, and
Vercel deployment.

> **Medical disclaimer:** This is an educational machine-learning
> demonstration, not a medical device, diagnosis, or treatment
> recommendation. Do not use the output to make medical decisions. Always
> consult a qualified healthcare professional.

## Live demo

Try the deployed browser application:

**[Open the Stroke Risk Predictor](https://stroke-prediction-mu.vercel.app/)**

The interface sends the selected inputs to the deployed prediction API and
displays the returned probability and the binary classification threshold.

## Highlights

- Complete notebook workflow: data loading, EDA, preprocessing, training,
  evaluation, visualization, and model export.
- Comparison of five classifiers:
  Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and
  Support Vector Machine.
- Class-weighted training for Logistic Regression, Decision Tree, Random
  Forest, and SVM.
- Stratified train/test split with a fixed random seed for reproducibility.
- Standardization of the `Age` feature using a saved `StandardScaler`.
- FastAPI endpoints for health checks and predictions with request validation.
- Browser UI with 16 model inputs, probability visualization, reset/randomize
  controls, and API connection status.
- Docker image and Vercel-compatible serverless deployment configuration.

## Model evaluation

The notebook uses an 80/20 stratified train/test split (`random_state=42`).
The table below is the recorded test-set output from the notebook. Models are
ranked by **Recall**, because missing a positive risk case is especially
important for this educational experiment.

| Model | Accuracy | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|
| **Support Vector Machine** | **98.81%** | **98.29%** | **99.08%** | **99.98%** |
| Gradient Boosting | 95.75% | 98.16% | 96.77% | 99.11% |
| Random Forest | 94.64% | 96.73% | 95.90% | 99.05% |
| Logistic Regression | 97.61% | 96.32% | 98.12% | 99.99% |
| Decision Tree | 88.02% | 91.65% | 90.85% | 86.47% |

The Support Vector Machine was selected by the notebook and saved as
`models/best_stroke_model.pkl`. The API uses a probability threshold of `0.5`
to convert the class-1 probability into the binary `prediction` value.

These metrics are not a clinical validation. They come from one hold-out
split, and the dataset's feature definitions and target construction should be
reviewed before any real-world use. High scores should not be interpreted as
evidence of medical accuracy or generalization to patients outside this data.

## Dataset

The notebook expects the public
[Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
at:

```text
data/raw/healthcare-dataset-stroke-data.csv
```

The raw dataset is intentionally not committed. See
[`data/README.md`](data/README.md) for download and placement instructions.

In the recorded notebook run, duplicate rows were removed and the resulting
data shape was **68,979 rows × 18 columns**. The notebook uses the binary
`At Risk (Binary)` target and drops the derived `Stroke Risk (%)` column before
training. Categorical `Absent`/`Present` values are mapped to `0`/`1`, while
the target is mapped from `Not At Risk`/`At Risk` to `0`/`1`.

The model receives these 16 features:

1. `Chest Pain`
2. `Shortness of Breath`
3. `Irregular Heartbeat`
4. `Fatigue & Weakness`
5. `Dizziness`
6. `Swelling (Edema)`
7. `Pain in Neck/Jaw/Shoulder/Back`
8. `Excessive Sweating`
9. `Persistent Cough`
10. `Nausea/Vomiting`
11. `High Blood Pressure`
12. `Chest Discomfort (Activity)`
13. `Cold Hands/Feet`
14. `Snoring/Sleep Apnea`
15. `Anxiety/Feeling of Doom`
16. `Age`

## How the project works

```text
Kaggle CSV
   │
   ▼
notebooks/01_stroke_risk_prediction.ipynb
   │  clean, encode, split, scale Age, train, compare
   ▼
models/best_stroke_model.pkl + models/scaler.pkl
   │
   ▼
src/main.py (FastAPI)
   ├── GET  /
   ├── GET  /health
   └── POST /predict
   │
   ▼
public/ (static browser application)
```

The API loads only the serialized model and scaler at startup. Training and
notebook visualization dependencies are not imported by the inference path.

## Project structure

```text
Stroke-Risk-Prediction/
├── README.md
├── requirements.txt              # API/runtime dependencies
├── requirements-dev.txt          # runtime + notebook/visualization tools
├── requirements-docker.txt       # lean Docker runtime dependencies
├── Dockerfile
├── vercel.json
├── notebooks/
│   └── 01_stroke_risk_prediction.ipynb
├── src/
│   ├── __init__.py
│   └── main.py                   # FastAPI app and validation
├── public/
│   ├── index.html                 # prediction form
│   ├── app.js                     # API integration and result handling
│   ├── styles.css
│   └── logo-animated.html
├── models/
│   ├── best_stroke_model.pkl
│   └── scaler.pkl
└── data/
    ├── README.md
    ├── raw/
    └── processed/
```

## Run locally

### 1. Install dependencies

For notebook work, model training, and the API:

```bash
python3 -m pip install -r requirements-dev.txt
```

For API-only local inference:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the FastAPI backend

From the repository root:

```bash
uvicorn src.main:app --reload
```

The backend runs at <http://127.0.0.1:8000>.

- Health check: <http://127.0.0.1:8000/health>
- Interactive OpenAPI docs: <http://127.0.0.1:8000/docs>
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`

### 3. Run the frontend

In a second terminal:

```bash
python3 -m http.server 5500 --directory public
```

Open <http://127.0.0.1:5500/>. The local frontend automatically uses
`http://127.0.0.1:8000` as its API base URL. On the Vercel deployment it uses
same-origin relative requests.

### 4. Reproduce the notebook workflow

1. Download the dataset from Kaggle.
2. Place it at
   `data/raw/healthcare-dataset-stroke-data.csv`.
3. Open
   [`notebooks/01_stroke_risk_prediction.ipynb`](notebooks/01_stroke_risk_prediction.ipynb)
   from the repository root.
4. Run the notebook cells in order.

The final cells write the trained classifier and age scaler to `models/`.

## API reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /predict`

The request must contain all 16 feature names. Binary features must be `0` or
`1`; `Age` must be an integer from `1` through `120`.

Example request:

```json
{
  "features": {
    "Chest Pain": 0,
    "Shortness of Breath": 0,
    "Irregular Heartbeat": 1,
    "Fatigue & Weakness": 0,
    "Dizziness": 0,
    "Swelling (Edema)": 0,
    "Pain in Neck/Jaw/Shoulder/Back": 0,
    "Excessive Sweating": 0,
    "Persistent Cough": 0,
    "Nausea/Vomiting": 0,
    "High Blood Pressure": 1,
    "Chest Discomfort (Activity)": 0,
    "Cold Hands/Feet": 0,
    "Snoring/Sleep Apnea": 0,
    "Anxiety/Feeling of Doom": 0,
    "Age": 54
  }
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.12,
  "threshold": 0.5
}
```

The API returns HTTP `422` for missing or unknown features, invalid binary
values, or an invalid age.

## Docker

The backend can run as a small API container using the runtime-only
dependencies:

```bash
docker pull mu3mar/stroke-risk-api:latest
docker run --rm --name stroke-risk-api -p 8000:8000 \
  mu3mar/stroke-risk-api:latest
```

Verify the container:

```bash
curl http://localhost:8000/health
```

Build the image locally if needed:

```bash
docker build -t stroke-risk-api .
docker run --rm -p 8000:8000 stroke-risk-api
```

The API documentation is available at <http://localhost:8000/docs>.

## Vercel deployment

The repository is configured for Vercel's native Python runtime:

- `src/main.py` exports the FastAPI instance as `app`.
- `requirements.txt` contains the inference dependencies.
- `vercel.json` bundles both files in `models/*.pkl`.
- `public/` is served as the static frontend.
- No build command or output directory is required.

To deploy, import the repository into Vercel with the repository root as the
project root. After deployment, verify `/`, `/health`, `/docs`, and the
browser form. Vercel functions are stateless and may cold-start while loading
the model.

## Limitations and responsible use

- This is a project/demo classifier, not a clinically validated system.
- The displayed probability is a model estimate, not a diagnosis or a
  patient's medical probability.
- Results can be affected by dataset bias, leakage, sampling, feature quality,
  and the single train/test split used in the notebook.
- The dataset and model should not be used for treatment, triage, or other
  high-impact decisions.
- Do not submit personally identifiable or sensitive health information to a
  public deployment.

## License and data attribution

The dataset is obtained separately from Kaggle and remains subject to its
source terms. Review the dataset page for attribution and usage conditions
before redistributing it.
