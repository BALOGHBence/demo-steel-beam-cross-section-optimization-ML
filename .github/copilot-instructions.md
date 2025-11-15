
# Copilot Instructions for AI Coding Agents

## Project Overview
This repository demonstrates ML-driven optimization of steel beam cross sections. It combines data generation, ML model training, and optimization workflows, primarily for civil/structural engineering applications.

## Architecture & Major Components
- **Data Generation**: `generate_learning_data.py` (root) creates training datasets from config files in `fixtures/`.
- **Model Training**: Jupyter notebooks in `src/` (e.g., `train_sklearn_utilization_estimators.ipynb`, `train_ANN_failure_predictor.ipynb`) train ML models for geometry validation, failure prediction, utilization estimation, and section property estimation.
- **Optimization**: Notebooks (`optimize_ml.ipynb`, `optimize_exact.ipynb`) use trained models to optimize beam cross sections, typically via genetic algorithms.
- **Core Library**: All reusable code is in the `src/cso/` package (logging, ML helpers, section calculations, PyTorch utilities, etc.).
- **Configuration**: Problem setup is via JSON config files in `fixtures/` (e.g., `config_rhs.json`).

## Developer Workflows
- **Install dependencies**: `uv pip install -e .` (see `pyproject.toml` for full list).
- **Generate data**: Run `uv run generate_learning_data.py --config fixtures/config_rhs.json --output fixtures/data_rhs_50000.csv [other args]`.
- **Train models**: Use the Jupyter notebooks in `src/`. MLflow is used for experiment tracking (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`).
- **Optimize**: Use the optimization notebooks in `src/`, referencing trained models and generated data.

## Project-Specific Patterns & Conventions
- **Notebook Naming**: Notebooks in `src/` are named by task (e.g., `train_ANN_failure_predictor.ipynb`), not numbered.
- **Config-Driven**: All major workflows are parameterized by JSON config files in `fixtures/`.
- **Data Files**: CSV files for training data are stored in `fixtures/`.
- **Logging**: Use `src/cso/logger.py` for consistent logging across scripts.
- **MLflow**: All model training should log experiments to MLflow for reproducibility.

## Integration Points & Dependencies
- **External Libraries**: Key dependencies include `sectionproperties`, `scikit-learn`, `torch`, `sigmaepsilon.solid.fourier`, `sigmaepsilon.math`, `mlflow`, and `papermill`.
- **ML Models**: Models are trained and saved via MLflow; ensure correct paths and experiment names.
- **Papermill**: Used for parameterized notebook execution.

## Examples
- **Generate Data**:
  ```sh
  uv run generate_learning_data.py --config fixtures/config_rhs.json --output fixtures/data_rhs_50000.csv --num_sections 500 --num_load_cases_per_section 100 --num_workers 8
  ```
- **Start MLflow UI**:
  ```sh
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```

## Key Files & Directories
- `generate_learning_data.py`: Data generation script (root)
- `src/cso/`: Core library modules (logger, ML helpers, section calculations)
- `src/`: Jupyter notebooks for model training and optimization
- `pyproject.toml`: Dependency and build configuration
- `fixtures/`: Problem configuration and generated datasets
- `mlruns/`: MLflow experiment tracking

---

**If any section is unclear or missing, please provide feedback for further refinement.**
