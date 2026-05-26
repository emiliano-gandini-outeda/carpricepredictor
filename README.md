# Car Price Predictor

Predict car prices from the classic Automobile Data Set using linear regression. End-to-end pipeline, REST API, and frontend, all containerized.

## Stack

- **src/**, Modular pipeline: ingestion, feature engineering, training, evaluation, prediction. Each step is a callable function, run them standalone or through `pipeline.py`.
- **api/**, FastAPI with separate routers for setup, evaluation, prediction, status, and notebook serving.
- **frontend/**, Plain HTML/CSS/JS SPA. Dropped in behind FastAPI's static file mount, talks to the API through `fetch()`. Notebooks rendered inline with marked + highlight.js.
- **Docker**, Slim Python image, uv for dependency management, compose volumes for data/models/reports.

## Quick Start

```bash
docker compose up --build
```

Opens at `http://localhost:8000`. The frontend has tabs for pipeline setup, evaluation, individual prediction, and browsing the analysis notebooks.

## Pipeline

```
data.py → features.py → train.py → evaluate.py → predict.py
```

Each module has a `run()` entry point. `pipeline.run_full_setup()` chains the whole thing. The API wraps these calls behind `/api/setup` and `/api/evaluate`.

## Project Structure

```
├── api/            # FastAPI application
├── frontend/       # SPA served by FastAPI
├── notebooks/      # EDA, feature analysis, training, evaluation
├── src/            # Pipeline modules
├── data/           # Raw and processed data (mounted volume)
├── models/         # Trained model, scaler, feature columns (mounted volume)
└── reports/        # Evaluation reports (mounted volume)
```

## Why No Framework

The frontend is a single HTML file with vanilla JS. No build step, no npm, no framework churn.

## Model Accuracy

Linear regression on 201 samples, 47 features. The model skews conservative, it's decent on mass-market cars, weaker on luxury outliers.

| Segment | Avg Error | Notes |
|---------|----------:|-------|
| Economy hatchbacks (20 cars) | 19.4% | Sweet spot. Cheap, common cars. |
| Mixed segment (20 cars) | 27.9% | Undershoots Jaguar, Mercedes, BMW by 40-60%. |

The economy set includes Civics, Corollas, Golfs, the kind of cars that populate the training data. The model systematically underprices luxury trims because there aren't enough of them in 201 rows to learn the premium.