# Stroke Risk Prediction System

## 🏥 Overview
This project is a comprehensive Machine Learning system designed to predict the risk of stroke in patients based on their medical history, demographic data, and current symptoms. It includes a complete pipeline from data analysis to a deployable web application.

## 📂 Project Structure

- **`stroke_risk_prediction.ipynb`**: The core analysis notebook. It handles data loading, extensive Exploratory Data Analysis (EDA), preprocessing, and trains 5 different Machine Learning models to find the best performer. It saves the best model and scaler to the `artifacts/` folder.
- **`api.py`**: A **FastAPI** backend that loads the trained model and serves predictions via a REST API.
- **`app.py`**: A **Streamlit** frontend that provides a user-friendly interface for doctors or patients to input symptoms and get a risk assessment.
- **`artifacts/`**: Directory where the trained model (`best_stroke_model.pkl`), scaler (`scaler.pkl`), and analysis charts are saved.
- **`stroke_dataset.csv`**: The dataset used for training and analysis.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training the Model
Before running the app, you need to train the model and generate the artifacts.

1. Open `stroke_risk_prediction.ipynb` in Jupyter Notebook or VS Code.
2. Run all cells.
3. This will create an `artifacts` folder containing the saved model (`.pkl` file) and various visualization charts.

### 3. Running the System
To use the application, you need to run both the backend API and the frontend interface.

**Step A: Start the API Server**
Open a terminal and run:
```bash
python3 api.py
```
*The API will start at `http://0.0.0.0:8000`.*

**Step B: Start the User Interface**
Open a **new** terminal window and run:
```bash
streamlit run app.py
```
*This will automatically open the application in your web browser (usually at `http://localhost:8501`).*

## 📊 Features
- **Deep EDA**: The notebook generates standardized, high-quality visualizations to understand risk factors.
- **Model Comparison**: Automatically compares Logistic Regression, Random Forest, SVM, Decision Tree, and Gradient Boosting.
- **Real-time Predictions**: The Streamlit app communicates with the API to provide instant results.
- **Risk Probability**: The app shows not just a binary result but the calculated probability of risk.

## ⚠️ Medical Disclaimer
This tool is for educational and demonstrative purposes only. It is **not** a diagnostic tool. Always consult a certified medical professional for health advice.
