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

## How to Run (Step-by-Step)

### Prerequisites

- A Snowflake account with `ACCOUNTADMIN` role (or a role with CREATE DATABASE/WAREHOUSE privileges)
- Access to **Cortex Code** — available in:
  - **Snowsight Notebooks** (browser-based, no setup needed)
  - **VS Code / Cursor** with the Snowflake extension

### Option A: Run via Snowsight Notebooks (Recommended)

1. **Open Snowsight** — navigate to your Snowflake account in the browser
2. **Open CoCo** — click the Cortex Code (AI) icon in the left sidebar of a Notebook
3. **Run Phase 1 (Setup):**
   - Create a new SQL Worksheet
   - Copy the **Phase 1** prompt from [`PROMPTS.md`](./PROMPTS.md) into CoCo's chat
   - CoCo will generate and execute SQL to create the database, warehouse, and 2 tables with synthetic data
   - Verify: `SELECT COUNT(*) FROM COCO_DS_DEMO.PUBLIC.TRANSACTIONS` should return 1,000,000
4. **Run Phase 2 (Fraud Detection):**
   - Create a new Notebook in Snowsight
   - Copy the **Phase 2** prompt from `PROMPTS.md` into CoCo's chat
   - CoCo generates the entire notebook — EDA, feature engineering, training, registry, scoring, monitoring
   - Click "Run All" or execute cells sequentially
5. **Run Phase 3 (Forecasting):**
   - Create another Notebook
   - Copy the **Phase 3** prompt into CoCo
   - Same flow — full regression pipeline with drift monitoring

### Option B: Run via VS Code / Cursor

1. **Install** the [Snowflake Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=snowflake.snowflake-vsc)
2. **Connect** to your Snowflake account via the extension
3. **Open Cortex Code** — use the CoCo chat panel in the extension
4. Follow the same Phase 1 → 2 → 3 prompt flow as above

### Option C: Run the Pre-built Notebooks Directly

If you just want to run the notebooks without CoCo generating them:

1. Complete Phase 1 setup (create DB + tables) by running the SQL from `PROMPTS.md` Phase 1
2. Upload `mlops_pipeline.ipynb` and `forecasting_pipeline.ipynb` to Snowsight Notebooks
3. Run all cells — they will execute against your `COCO_DS_DEMO` database

### Tips

- **Runtime:** Phase 2 takes ~3-5 minutes to run (training on 1M rows). Phase 3 is faster (~1 min).
- **Warehouse size:** Medium is sufficient. Increase to Large if training is slow.
- **CoCo generation:** Each prompt typically produces the full notebook in 30-60 seconds.
- **Re-running:** The notebooks are idempotent — they overwrite tables/models on re-run.
- **Monitoring Task:** The created Task runs daily at 6 AM UTC. Suspend it after the demo with:
  ```sql
  ALTER TASK COCO_DS_DEMO.PUBLIC.DAILY_MODEL_MONITORING_TASK SUSPEND;
  ```

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
