# Cortex Code MLOps Demo

End-to-end MLOps demo built entirely with **Snowflake Cortex Code (CoCo)** — demonstrating fraud detection and revenue forecasting pipelines using traditional ML with Snowflake-native MLOps components.

## What This Demonstrates

A data scientist can build a complete ML lifecycle using only natural language prompts in Cortex Code:

- **EDA** — Exploratory data analysis with visualizations
- **Feature Engineering** — Statistical, temporal, and encoded features in pandas
- **Model Training** — scikit-learn/XGBoost model comparison and selection
- **Model Registry** — Register models with `snowflake.ml.registry`
- **Warehouse Inference** — Serve predictions via `mv.run()` and `MODEL!PREDICT()` SQL syntax
- **Batch Scoring** — Score datasets and write predictions to Snowflake tables
- **Monitoring & Drift** — KS test-based data drift, performance tracking, prediction drift
- **Automation** — Snowflake Tasks (scheduled monitoring) + Alerts (threshold notifications)

## Files

| File | Description |
|------|-------------|
| `PROMPTS.md` | Copy-paste prompts for Cortex Code to recreate the full demo |
| `mlops_pipeline.ipynb` | Phase 2 — Fraud detection MLOps (classification) |
| `forecasting_pipeline.ipynb` | Phase 3 — Revenue forecasting MLOps (regression) |

## How to Use

1. Open **Cortex Code** in Snowflake (Snowsight Notebooks or VS Code)
2. Copy the **Phase 1** prompt from `PROMPTS.md` — this creates the database, warehouse, and synthetic data
3. Copy the **Phase 2** prompt — CoCo generates the full fraud detection notebook
4. Copy the **Phase 3** prompt — CoCo generates the revenue forecasting notebook

Each prompt produces a complete, runnable notebook in one shot.

## Architecture

```
TRANSACTIONS (1M rows)          DAILY_REVENUE (5 years)
        |                                |
    [EDA + Feature Eng]             [EDA + Feature Eng]
        |                                |
    [XGBoost / RF / LR]            [XGBoost / RF / Ridge]
        |                                |
    [Snowflake Model Registry]     [Snowflake Model Registry]
        |                                |
    [Warehouse Inference]          [Warehouse Inference]
        |                                |
    [FRAUD_PREDICTIONS]            [REVENUE_FORECAST]
        |                                |
    [Drift Detection + KS Test]    [Drift Detection + KS Test]
        |                                |
    [Task: Daily Monitoring]       [Task: Daily Monitoring]
        |                                |
    [Alert: Drift Notification]    [Alert: MAPE Threshold]
```

## Tech Stack

- **ML:** scikit-learn, XGBoost, scipy (KS test)
- **Data:** Snowpark (data loading), pandas (feature engineering)
- **MLOps:** Snowflake Model Registry, warehouse inference (`target_platforms=["WAREHOUSE"]`)
- **Automation:** Snowflake Tasks, Snowflake Alerts
- **Visualization:** matplotlib, seaborn

## Key Design Decisions

- **Warehouse inference only** — no compute pools or SPCS (`create_service()` is not used)
- **No Feature Store** — features computed inline with pandas for simplicity
- **Stratified splits** — preserves 3% fraud rate across train/test
- **Class imbalance handling** — `scale_pos_weight` (XGBoost), `class_weight='balanced'` (RF/LR)
- **Temporal split for forecasting** — last 30 days held out, no shuffle

## Snowflake Objects Created

```
Database:   COCO_DS_DEMO
Warehouse:  COCO_DS_WH (Medium, auto-suspend 60s)
Schema:     PUBLIC

Tables:     TRANSACTIONS, DAILY_REVENUE, FRAUD_PREDICTIONS,
            REVENUE_FORECAST, MODEL_MONITORING_LOG, FORECAST_MONITORING_LOG

Models:     FRAUD_DETECTOR (v1), REVENUE_FORECASTER (v1)

Procedures: RUN_MODEL_MONITORING()
Tasks:      DAILY_MODEL_MONITORING_TASK (6:00 AM UTC)
Alerts:     MODEL_DRIFT_ALERT (6:30 AM UTC)
```

## Cleanup

```sql
DROP DATABASE IF EXISTS COCO_DS_DEMO;
DROP WAREHOUSE IF EXISTS COCO_DS_WH;
```
