# ML Model API

This repository provides a minimal machine learning service that supports:

- Training a binary classification model using scikit-learn
- Saving trained model artifacts
- Running predictions via an API

## Current Features
- Logistic Regression training
- Model persistence
- Prediction endpoint

## Planned Extensions
- Model explainability (SHAP-based global and local explanations)
- Partial dependence plots
- Explainability report generation

## Running with Docker

Build and run the container:

```bash
docker build -t ml-explainability-pack .
docker run -p 8000:8000 ml-explainability-pack
```

Open http://localhost:8000 in your browser to access the UI.
