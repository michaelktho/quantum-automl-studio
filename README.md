# Quantum AutoML Studio

**A light, minimal, scientific Streamlit application for quantum machine learning on CSV datasets.**

Upload a CSV, pick a target, configure a reproducible train–test split and compare an exact state-vector quantum model with a classical support vector baseline. The first release is designed for CPU-based experiments on free Streamlit Community Cloud infrastructure.

## Scope

| Task | Quantum estimator | Classical baseline |
|---|---|---|
| Binary classification | Trainable variational quantum classifier | RBF SVC |
| Binary classification | Fidelity-based quantum kernel SVM | RBF SVC |
| Regression | Trainable variational quantum regressor | RBF SVR |

Features include: CSV upload, example datasets, missing-data imputation, categorical one-hot encoding, standardisation, PCA where necessary, quantum angle encoding, configurable stratified train–test splitting for classification, held-out metrics and Plotly visualisations, CSV prediction export and JSON configuration export.

**Scientific caveat:** This application is a small-scale CPU simulation, not a quantum computer or a demonstration of quantum advantage. Quantum and classical models are compared using the same held-out split. No hyperparameter search or repeated cross-validation is included; a single test split is not sufficient for strong research claims.

## Installation

Use **Python 3.12**, preferably in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The package ranges in `requirements.txt` are constrained for major-version compatibility, not a reproducible lockfile. For a stable production release, test and pin exact working versions after a complete cloud build.

## Deployment on Streamlit Community Cloud

1. Create a GitHub repository and upload **the complete project**, preserving the `.streamlit/config.toml` path.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/), select **Create app** and connect the repository.
3. Choose your branch and set the entrypoint to **`app.py`**.
4. In advanced deployment settings, select **Python 3.12**.
5. Deploy. Streamlit will install packages from `requirements.txt`.
6. Verify with the bundled example data, then test both learning tasks and the CSV/JSON exports.

Public deployment is free under the provider's current eligibility, resource limits and usage policies, which can change. CPU-intensive quantum optimisation may be slow or unavailable under load. This project does not require credentials or external quantum hardware.

## Working with your own dataset

- CSV files only, up to 5 MB and 80 columns.
- At least 12 usable labelled rows. Up to 500 are sampled reproducibly (configurable within the app).
- Binary classification must have exactly two classes, each with at least two examples; regression must have a numeric target.
- Choose a target and test split (10–40%). Classification uses stratified splitting when feasible.
- The model reduces encoded features to at most 2–4 qubits. PCA and scalers are **fitted on the training partition only**.
- For the quantum kernel SVM, limit the train partition to 180 rows because the fidelity matrix is quadratic in dataset size.
- Quantum kernel results may require more compute than the classical baseline. Variational training can be slow even on 120 rows and 15 epochs.

## Source tree

```text
app.py                        Streamlit page, controls, training and results
models/quantum.py             PennyLane variational models and quantum kernel SVM
utils/preprocessing.py        Split, imputation, encoding, PCA and scaling
utils/evaluation.py           Classical baselines, metrics and prediction exports
utils/visualisation.py        Plotly charts
sample_data/                  Example regression and classification CSVs
tests/                        Pipeline and quantum smoke tests
.streamlit/config.toml        Minimal light theme and upload settings
requirements.txt              Application dependencies
```

## Tests

```bash
python -m pip install pytest
python -m pytest -q
python -m compileall -q app.py models utils tests
```

The quantum tests execute only when PennyLane is installed. If the installation is missing, pytest skips those tests. Always run the entire test suite after installing from `requirements.txt` and before public deployment.

## Design and privacy notes

The UI uses a bright neutral background, restrained scientific blue accents, responsive columns, legible charts and minimal animation. Uploaded CSVs are parsed in the app's active Python session rather than written to a persistent database. **This is not a security guarantee:** the hosting platform handles memory, logs and service infrastructure. Do not upload private, sensitive or confidential datasets to a public demonstration. No user accounts, rate limiting, authentication or persistent prediction history are included.

## Limitations / future work

- No multiclass support, time-series aware split, external prediction-file workflow or model serialisation yet.
- No early stopping or automated tuning; both could reduce compute or improve performance.
- Test scores are displayed without confidence intervals. For research use, add repeated splits or cross-validation confined to the training data and retain an untouched final test set.
- Users should avoid target leakage through predictive feature columns, or dependent rows. The app cannot automatically detect these issues.
- Kernel model circuit visuals are not displayed; variational circuits have an ASCII circuit view.
- The classical baseline shares reduced, angle-scaled features with the quantum model for a controlled input comparison, which does not represent a separately tuned classical optimum.
- Streamlit's synchronous execution is not a job queue. Busy multi-user deployments need resource management and potentially background workers.

## Sources

- [PennyLane quantum kernel matrix documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.kernels.square_kernel_matrix.html)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit dependency management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

MIT-licensed demonstration. See `LICENSE`.
