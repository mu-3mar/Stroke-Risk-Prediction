# Dataset

## Overview

This project uses the **Stroke Prediction Dataset**, which contains demographic, health-related, and lifestyle information for predicting the likelihood of a stroke.

The dataset includes 5,110 records and 11 predictive features, along with a unique identifier and a binary target variable (`stroke`).

## Dataset Source

The dataset is publicly available on Kaggle:

**[Stroke Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**

## Expected location

Place the downloaded CSV at:

```text
data/raw/healthcare-dataset-stroke-data.csv
```

The `data/raw/` directory is ignored by git so the dataset stays local.

## Download Instructions

1. Visit the Kaggle dataset page using the link above.
2. Sign in to your Kaggle account.
3. Navigate to the dataset page and click **Download**.
4. Extract the downloaded ZIP file.
5. Place the CSV file (`healthcare-dataset-stroke-data.csv`) in `data/raw/`.

## Notes

- The dataset is not included in this repository.
- Users must download the dataset directly from Kaggle before running the project notebook.
- The target variable is `stroke`, where `1` indicates a stroke event and `0` indicates no stroke event.