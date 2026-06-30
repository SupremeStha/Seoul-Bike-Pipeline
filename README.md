# Seoul Bike Sharing Demand — MLOps Pipeline

An end-to-end MLOps pipeline that predicts Seoul bike-sharing rental demand, built around a fully containerized workflow for data ingestion, validation, preprocessing, model training, deployment, and monitoring.

## Overview

This project automates the full lifecycle of a machine learning model for forecasting bike rental demand in Seoul, using historical weather and rental data. It's orchestrated end-to-end with Apache Airflow, with data quality checks, experiment tracking, a served prediction API, and automated drift/performance monitoring.

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Data storage | MariaDB ColumnStore |
| Caching / messaging | Redis |
| Data validation | Great Expectations |
| Experiment tracking & model registry | MLflow |
| Model serving | FastAPI |
| Monitoring (drift & performance) | Evidently AI |
| Containerization | Docker Compose |

## Pipeline Stages

The Airflow DAG (`bike_pipeline_dag.py`) orchestrates the following stages:

1. **Ingest** (`tasks_ingest.py`) — Pulls raw bike-sharing data into the pipeline.
2. **Validate** (`tasks_validate.py`) — Runs data quality checks with Great Expectations.
3. **Preprocess** (`tasks_preprocess.py`) — Cleans and transforms features for training.
4. **Load to DB** (`tasks_load_db.py`) — Persists processed data into MariaDB ColumnStore.
5. **Train** (`tasks_train.py`) — Trains the regression model and logs experiments/artifacts to MLflow.
6. **Deploy** (`deploy.py`) — Registers/promotes the trained model for serving.
7. **Monitor** (`monitor.py`) — Generates Evidently AI reports for data drift and regression performance.

A FastAPI service (`app.py`) exposes the trained model for real-time predictions.

## Project Structure

```
.
├── dags/
│   ├── bike_pipeline_dag.py    # Main Airflow DAG definition
│   ├── tasks_ingest.py         # Data ingestion task
│   ├── tasks_validate.py       # Data validation (Great Expectations)
│   ├── tasks_preprocess.py     # Feature preprocessing
│   ├── tasks_load_db.py        # Load processed data into MariaDB
│   ├── tasks_train.py          # Model training + MLflow logging
│   ├── deploy.py               # Model deployment/registration
│   ├── monitor.py              # Drift & performance monitoring
│   ├── config.py               # Shared configuration
│   └── app.py                  # FastAPI prediction service
├── data/
│   └── SeoulBikeData.csv       # Raw dataset (not tracked in git)
├── reports/
│   ├── data_drift_report.html
│   └── regression_performance_report.html
├── docker-compose.yml
├── Dockerfile.airflow
├── Dockerfile.fastapi
├── Dockerfile.mcs              # MariaDB ColumnStore image
├── EDA.ipynb                   # Exploratory data analysis notebook
└── requirements.txt
```

## Getting Started

### Prerequisites
- Docker & Docker Compose
- The `SeoulBikeData.csv` dataset placed in `data/`

### Run the pipeline

```bash
docker compose up --build
```

This spins up Redis, MariaDB ColumnStore, Postgres (Airflow metadata DB), Airflow (webserver + scheduler + MLflow UI), and the FastAPI prediction service.

### Access the services

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:9090 | `admin` / `admin123` |
| MLflow UI | http://localhost:5000 | — |
| FastAPI prediction service | http://localhost:8000 | — |

### Trigger the pipeline

The pipeline DAG (`seoul_bike_pipeline`) is automatically unpaused and triggered on startup via the `airflow-auto-trigger` service. You can also trigger it manually from the Airflow UI.

## Monitoring

After each pipeline run, Evidently AI generates two HTML reports in `reports/`:
- **`data_drift_report.html`** — flags distributional shifts between reference and current data.
- **`regression_performance_report.html`** — tracks model performance metrics over time.

## Notes

- Credentials in `docker-compose.yml` are local development defaults, not intended for production use.
- This project was developed as part of CMP5366 (Data Management and MLOps) coursework.
