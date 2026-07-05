# Cortex Code Data Science Demo - Prompts

Financial fraud detection + revenue forecasting with full MLOps. Copy-paste each prompt into Cortex Code.

---

## Phase 1: Environment Setup

```
Create a demo environment in Snowflake for a financial data science showcase:

1. Create database COCO_DS_DEMO with a single PUBLIC schema.
2. Create warehouse COCO_DS_WH (size Medium, auto-suspend 60s, auto-resume true).

3. Create these tables with synthetic data:

   a) COCO_DS_DEMO.PUBLIC.TRANSACTIONS (1000000 rows, ~3% fraud rate)
      - transaction_id, customer_id, transaction_date, amount, merchant_category
        (grocery/electronics/travel/dining/fuel/entertainment), payment_method
        (credit_card/debit_card/wire/digital_wallet), channel (online/in_store/atm),
        location_country, is_international (boolean), hour_of_day, is_fraud (boolean)
      - Fraud patterns: higher amounts, unusual hours (1-5am), more international,
        skewed toward wire/digital_wallet, clustered in travel/electronics.

   b) COCO_DS_DEMO.PUBLIC.DAILY_REVENUE (5 years daily)
      - date, revenue, channel (online/in_store/atm)
      - Weekly seasonality, upward trend, 5-8 anomalous spikes/drops.
        Online ~50%, retail ~35%, atm ~15%.

Execute all SQL. Verify with SELECT COUNT(*) on each table.
```

---

## Phase 2: Full Classification Pipeline (Fraud Detection)

```
Create a notebook "mlops_pipeline.ipynb" that demonstrates a complete MLOps lifecycle for fraud detection using scikit-learn/XGBoost — from EDA through model serving — all within Snowflake:

EXPLORATORY DATA ANALYSIS:
- Connect via Snowpark, load COCO_DS_DEMO.PUBLIC.TRANSACTIONS into pandas
- Basic stats: row count, fraud rate, missing values
- Visualizations: fraud rate by merchant_category, by payment_method, by channel,
  amount distribution (fraud vs legit), fraud by hour_of_day, fraud vs is_international
- Correlation heatmap of numeric features with is_fraud

FEATURE ENGINEERING (pandas):
- AMOUNT_ZSCORE: standardized transaction amount
- IS_HIGH_AMOUNT: amount > 2 standard deviations above mean
- IS_ODD_HOUR: hour_of_day between 1 and 5
- CATEGORY_RISK_SCORE: historical fraud rate per merchant_category (target encoding)
- CHANNEL_RISK_SCORE: historical fraud rate per channel
- One-hot encode: merchant_category, payment_method, channel
- Drop non-feature columns (transaction_id, customer_id, transaction_date, location_country)

MODEL TRAINING (scikit-learn):
- Train/test split (70/30, stratified)
- Train XGBClassifier (handle imbalance with scale_pos_weight)
- Also train a RandomForestClassifier and LogisticRegression for comparison
- Evaluate all three: accuracy, precision, recall, F1, AUC-ROC
- Plot ROC curves overlaid, confusion matrix for best model
- Feature importance (top 10) for best model
- Select best model based on F1/AUC

MODEL REGISTRY & WAREHOUSE INFERENCE:
- Set session database context with USE DATABASE COCO_DS_DEMO and USE SCHEMA PUBLIC
  before registry operations
- Log best model to Snowflake Model Registry (snowflake.ml.registry.Registry)
  with name "fraud_detector", version "v1", metrics, sample_input_data,
  and target_platforms=["WAREHOUSE"]
- Show how to list registered models and versions
- Demonstrate warehouse-based inference using mv.run(data, function_name="predict_proba")
  — this runs on the warehouse directly, no create_service() needed
- Show the SQL syntax for calling the model: MODEL_NAME!PREDICT_PROBA(...)
- Note: create_service() is for SPCS container deployments only (requires
  service_compute_pool), NOT for warehouse inference

BATCH SCORING:
- Score all transactions using mv.run() or best_model.predict_proba() on the full
  feature matrix
- Write predictions to COCO_DS_DEMO.PUBLIC.FRAUD_PREDICTIONS
  (transaction_id, fraud_probability, predicted_fraud, scored_at)
- Summary: count by risk tier (high >0.8, medium 0.4-0.8, low <0.4)
- Top 20 most suspicious transactions

MODEL MONITORING & DRIFT (automated with Tasks + Alerts):
- Simulate a "new batch" of transactions with shifted distributions
  (increase fraud rate to 5%, shift amount higher, more digital_wallet)
- Score the new batch with the model
- PERFORMANCE MONITORING: calculate precision/recall on new batch, compare vs
  training metrics, flag if F1 drops below threshold
- DATA DRIFT: compare feature distributions (training vs new batch) using PSI
  or KS test for each feature. Visualize drift as bar chart.
- PREDICTION DRIFT: compare prediction distribution shift
- Save monitoring results to COCO_DS_DEMO.PUBLIC.MODEL_MONITORING_LOG
  (run_date, metric_name, training_value, current_value, drift_score, alert_flag)
- Create a Snowflake TASK that runs monitoring daily (scheduled stored procedure
  that scores new data, computes drift metrics, writes to monitoring log)
- Create a Snowflake ALERT that fires when drift_score exceeds threshold
  or F1 drops below acceptable level

Include markdown functional explanations for each cell in sections.
```

---

## Phase 3: Full Regression Pipeline (Revenue Forecasting)

```
Create a notebook "forecasting_pipeline.ipynb" that demonstrates a complete MLOps lifecycle for revenue forecasting using scikit-learn/XGBoost — mirroring the same structure as Phase 2:

EXPLORATORY DATA ANALYSIS:
- Connect via Snowpark, load COCO_DS_DEMO.PUBLIC.DAILY_REVENUE into pandas
- Basic stats: date range, row count, revenue stats per channel
- Visualizations: revenue over time by channel (line chart), average revenue by day-of-week
  (weekly seasonality), monthly trend, distribution of revenue values per channel
- Identify visible anomalies or trend breaks

FEATURE ENGINEERING (pandas):
- Lag features: revenue_lag_7, revenue_lag_14, revenue_lag_30
- Rolling averages: rolling_7d_avg, rolling_30d_avg
- Calendar features: day_of_week, month, is_weekend, is_month_start, is_month_end
- Channel one-hot encoding
- Target: next-day revenue
- Drop rows with NaN from lag/rolling calculations

MODEL TRAINING (scikit-learn):
- Train/test split: hold out last 30 days as test set (temporal split, no shuffle)
- Train XGBRegressor, RandomForestRegressor, and Ridge for comparison
- Evaluate all three: MAE, RMSE, MAPE, R²
- Plot predicted vs actual for each model on test set
- Feature importance (top 10) for best model
- Select best model based on MAPE/R²

MODEL REGISTRY & WAREHOUSE INFERENCE:
- Set session database context with USE DATABASE COCO_DS_DEMO and USE SCHEMA PUBLIC
  before registry operations
- Log best model to Snowflake Model Registry (snowflake.ml.registry.Registry)
  with name "revenue_forecaster", version "v1", metrics, sample_input_data,
  and target_platforms=["WAREHOUSE"]
- Show how to list registered models and versions
- Demonstrate warehouse-based inference using mv.run(data, function_name="predict")
  — this runs on the warehouse directly, no create_service() needed
- Show the SQL syntax for calling the model: MODEL_NAME!PREDICT(...)
- Note: create_service() is for SPCS container deployments only (requires
  service_compute_pool), NOT for warehouse inference

BATCH SCORING (FORECAST GENERATION):
- Use mv.run() to generate next-30-day revenue forecast
- Write predictions to COCO_DS_DEMO.PUBLIC.REVENUE_FORECAST
  (date, channel, predicted_revenue, scored_at)
- Compare forecast vs held-out actuals: MAPE per channel, plot with confidence
- Plot full history + 30-day forecast

MODEL MONITORING & DRIFT (automated with Tasks + Alerts):
- Simulate "next week" actual revenue arriving with shifted distribution
- Score the new period with the model
- PERFORMANCE MONITORING: calculate MAPE on new actuals, compare vs
  training MAPE, flag if degradation exceeds threshold
- DATA DRIFT: compare distribution of incoming revenue features vs training
  data using PSI or KS test. Visualize drift as bar chart.
- PREDICTION DRIFT: compare prediction distribution shift
- Save monitoring results to COCO_DS_DEMO.PUBLIC.FORECAST_MONITORING_LOG
  (run_date, channel, metric_name, training_value, current_value, drift_score, alert_flag)
- Create a Snowflake TASK that runs monitoring daily (scheduled stored procedure
  that scores new data, computes drift metrics, writes to monitoring log)
- Create a Snowflake ALERT that fires when MAPE exceeds threshold
  or drift detected

Include markdown functional explanations for each cell in sections.
```

---

## Quick Reference

| Phase | Creates | Demo Point |
|-------|---------|------------|
| 1 | DB + warehouse + 2 tables | "One prompt sets up everything" |
| 2 | Fraud detection notebook: EDA → Feature Store → training → registry → warehouse service → scoring → monitoring → Tasks + Alerts | "Full MLOps with Snowflake-native components" |
| 3 | Forecasting notebook: EDA → features → scikit-learn/XGBoost → registry → warehouse inference → forecast → monitoring → Tasks + Alerts | "Regression MLOps lifecycle, same pattern" |

---

## Cleanup

```
DROP DATABASE IF EXISTS COCO_DS_DEMO;
DROP WAREHOUSE IF EXISTS COCO_DS_WH;
```
