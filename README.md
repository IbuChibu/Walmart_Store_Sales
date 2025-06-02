# Walmart Weekly Sales Forecasting with Machine Learning

This project leverages historical sales, economic indicators, and seasonal features to predict Walmart's future weekly sales. The goal is to support **data-driven markdown strategies**, inventory planning, and improve operational decisions using interpretable machine learning models.

---

## Project Overview

- **Objective**: Forecast weekly sales across Walmart stores using historical data and economic trends.
- **Key Deliverables**:
  - Predictive model for `Future_Weekly_Sales_t1` (1-week ahead)
  - Visual evaluation of predictions
  - Economic impact analysis on sales
  - Business recommendations for markdown and inventory optimization

---

## Features Used

- **Historical Sales**:
  - `Weekly_Sales`, `Lag_1` to `Lag_4`, `MA_4`, `MA_8`
- **Seasonality**:
  - `Month`, `Holiday_Flag`
- **Economic Indicators**:
  - `CPI`, `Unemployment`, `Fuel_Price`

---

## Model Development Pipeline

### Data Preparation
- Preprocessing pipeline with `StandardScaler` (to prevent data leakage)
- Time-based train-test split (80/20, no shuffle)
- Feature engineering: lag features, moving averages

### Model Selection
- Baseline models:
  - Linear Regression
  - Random Forest
  - XGBoost
- Evaluation via 5-fold cross-validation using **negative RMSE**

### Hyperparameter Tuning
- **Model chosen**: `XGBoost Regressor` (best base performance)
- Optimization method: `RandomizedSearchCV` for faster tuning
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.

---

## 🧪 Model Evaluation

| Metric | Value |
|--------|-------|
| **R² Score** | 0.968 |
| **RMSE** | ~70,356 |
| **MAE** | Reported |

### Visualizations
- Predicted vs Actual Scatter Plot
- Store-level forecast (Store 39)
- Feature Importance Bar Chart (XGBoost - Gain)

---

## Feature Importance

Top contributing features:
- `MA_4`, `MA_8`, `Weekly_Sales`, `Lag_1`, `Lag_2`
- Seasonality (`Month`, `Holiday_Flag`)
- Lower contribution from CPI, Fuel Price, Unemployment

---

## Business Insights & Recommendations

### Economic Indicators

| Feature        | MI Score | Pearson Corr | Interpretation |
|----------------|----------|----------------|----------------|
| Unemployment   | 0.689    | -0.106         | Strongest non-linear predictor |
| CPI (Inflation)| 0.521    | -0.073         | Moderate influence |
| Fuel Price     | 0.056    | +0.009         | Minimal direct impact |

- **Implication**: Tailor pricing/markdowns in response to unemployment and inflation, not fuel prices directly.

### Markdown Strategy Optimization
- Use model outputs to **time markdowns** pre-season (e.g. Nov–Dec)
- Apply **targeted discounts** during economic downturns (high CPI or unemployment)
- **Pre-emptive markdowns** before demand drops (based on predictions)

---

## Deployment Strategy

1. **Data Pipeline**:
   - Schedule ingestion via Airflow or AWS Step Functions
   - Automate feature engineering with Spark/Dask

2. **Model API**:
   - Serve with FastAPI or Flask
   - Docker container for portability
   - Deploy on AWS Lambda / ECS

3. **Monitoring**:
   - Use MLflow or Weights & Biases for tracking
   - A/B test predictions vs actual sales
   - Retrain periodically (weekly/monthly)

---

## Future Work

- Switch to **Bayesian Optimization** for hyperparameter tuning
- Try **LightGBM** or **LSTM** for sequential modeling
- Include **real-time economic/weather forecasts**
- Train on **daily-level** sales data for finer granularity

---

## Tech Stack

- **Languages**: Python
- **Libraries**: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost
- **Tools**: Jupyter Notebook, matplotlib

---

## Author Notes

This project shows how ML can support business strategy in retail using interpretable forecasting. It's a strong foundation for further enhancements like deep learning or integration with real-time APIs.
