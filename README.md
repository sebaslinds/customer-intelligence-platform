# Customer Intelligence Platform

A production-ready data engineering project template for building customer analytics pipelines with Python, Snowflake, dbt, and a lightweight API layer.

## Overview

Customer Intelligence Platform is designed to ingest raw customer data, transform it into analytics-ready models, and expose curated customer metrics for downstream dashboards, machine learning, and operational applications.

The repository is organized for a GitHub portfolio project while keeping production conventions in place: environment-based configuration, structured logging, modular Python packages, dbt transformations, and Snowflake connectivity.

## Architecture

```text
customer-intelligence-platform/
  ingestion/              Python ingestion jobs and source connectors
  transformations/dbt/    dbt project for Snowflake transformations
  ml/                     Feature engineering and modeling workflows
  api/                    FastAPI service for curated customer insights
  dashboard/              Dashboard assets and BI integration notes
  data/raw/               Local raw-data landing area for development only
  config/                 Application settings and logging configuration
  scripts/                Operational scripts for local development
```

## Features

- Python backend with FastAPI-ready application structure
- Snowflake support through SQLAlchemy and the Snowflake connector
- dbt project scaffold using the Snowflake adapter
- `.env` based configuration with a committed `.env.example`
- Centralized structured logging
- Modular folders for ingestion, transformations, ML, API, and dashboard work
- Clean `.gitignore` for Python, dbt, local data, secrets, and virtual environments

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional workflow-specific dependencies are split out to keep lightweight deployments, such as Streamlit Cloud, from installing services they do not need:

```bash
pip install -r requirements-dbt.txt
pip install -r requirements-ingestion.txt
pip install -r requirements-streaming.txt
pip install -r requirements-data-quality.txt
```

### 3. Configure environment variables

Copy the example file and fill in your Snowflake credentials:

```bash
cp .env.example .env
```

### 4. Run the API locally

```bash
uvicorn api.main:app --reload
```

### 5. Run dbt

```bash
cd transformations/dbt
dbt debug
dbt run
dbt test
```

## Snowflake Configuration

The application reads Snowflake settings from environment variables:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`

The dbt project uses the `customer_intelligence` profile. A starter profile template is included at `transformations/dbt/profiles.yml.example`.

## Development Notes

- Keep raw files in `data/raw/` for local testing only. Production raw data should land in object storage or Snowflake stages.
- Do not commit `.env`, credentials, dbt targets, local datasets, or generated artifacts.
- Place reusable source connectors in `ingestion/`.
- Place warehouse transformations in `transformations/dbt/models/`.
- Place reusable model training and scoring code in `ml/`.

## Deployment

### Render API

The FastAPI service is Docker-ready. Render can deploy it with the included `Dockerfile` and `render.yaml`.

Required Render environment variables:

- `APP_ENV=production`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `OPENAI_API_KEY`
- `MODEL_PATH`

Render provides `PORT` automatically. The container starts with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Streamlit Cloud Dashboard

Deploy the dashboard with:

- Main file path: `dashboard/app.py`
- Python dependencies: `dashboard/requirements.txt`
- Python runtime: select Python 3.11 in Streamlit Cloud advanced settings
- Secrets: copy values from `.streamlit/secrets.toml.example` into Streamlit Cloud secrets

Streamlit Cloud exposes secrets as environment variables, which are read by the shared settings layer.

### Dependency Locking

`requirements.txt` contains pinned deploy dependencies. `requirements.lock` mirrors the frozen application dependency set for deployment review.

## Portfolio Value

This project demonstrates practical data engineering patterns:

- Data ingestion with Python
- Warehouse-first transformation with dbt
- Snowflake-backed analytics modeling
- API delivery of customer intelligence metrics
- Production-minded configuration, logging, and repository hygiene
