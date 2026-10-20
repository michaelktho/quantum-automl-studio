"""Leakage-safe tabular preparation for small quantum simulators."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder

MAX_ROWS = 500
MAX_COLUMNS = 80

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    test_indices: np.ndarray
    feature_names: list
    label_encoder: object
    preprocessing: object
    n_train: int
    n_test: int


def read_csv(file):
    df = pd.read_csv(file, low_memory=False)
    if df.empty or len(df.columns) < 2:
        raise ValueError('CSV must contain at least one observation and two columns.')
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f'Please upload a dataset with at most {MAX_COLUMNS} columns.')
    if df.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    return df


def prepare(df, target, task, qubits=2, test_size=.2, seed=42, scaler='Standard', sample_limit=200):
    if target not in df.columns:
        raise ValueError('Select an existing target column.')
    if task not in ('Classification', 'Regression'):
        raise ValueError('Unsupported learning task.')
    if not 0.1 <= test_size <= 0.4:
        raise ValueError('Test fraction must be between 0.1 and 0.4.')
    if not 2 <= qubits <= 4:
        raise ValueError('Only 2–4 qubits are supported in this release.')
    data = df.dropna(subset=[target]).copy()
    if task == 'Regression':
        data[target] = pd.to_numeric(data[target], errors='coerce')
        data = data.dropna(subset=[target])
    if len(data) < 12:
        raise ValueError('At least 12 valid labelled observations are required.')
    if len(data) > sample_limit:
        data = data.sample(n=sample_limit, random_state=seed).sort_index()
    X = data.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)
    if X.shape[1] < 1:
        raise ValueError('At least one feature column is required.')
    y_raw = data[target]
    encoder = None
    if task == 'Classification':
        if y_raw.nunique() != 2:
            raise ValueError('Version 1 supports binary classification only (exactly two target classes).')
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        counts = np.bincount(y)
        if np.min(counts) < 2:
            raise ValueError('Each target class must have at least two observations.')
        stratify = y
    else:
        y = y_raw.to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError('Regression targets must be finite numbers.')
        stratify = None
    try:
        train_pos, test_pos = train_test_split(np.arange(len(X)), test_size=test_size,
                                                random_state=seed, stratify=stratify)
    except ValueError as exc:
        raise ValueError(f'Cannot create this split: {exc}') from exc
    X_train_raw, X_test_raw = X.iloc[train_pos], X.iloc[test_pos]
    numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    if not numeric_cols and not categorical_cols:
        raise ValueError('No supported feature columns found.')
    branches = []
    if numeric_cols:
        branches.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                              ('scale', StandardScaler())]), numeric_cols))
    if categorical_cols:
        branches.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols))
    column_transformer = ColumnTransformer(branches, sparse_threshold=0)
    # All fit operations occur after the split, on training rows only.
    Xtr = column_transformer.fit_transform(X_train_raw)
    Xte = column_transformer.transform(X_test_raw)
    if Xtr.shape[1] == 0:
        raise ValueError('Preprocessing produced no usable features.')
    n_components = min(qubits, Xtr.shape[0], Xtr.shape[1])
    pca = PCA(n_components=n_components, random_state=seed) if Xtr.shape[1] > qubits else None
    if pca is not None:
        Xtr, Xte = pca.fit_transform(Xtr), pca.transform(Xte)
    # Fit angle scaling ONLY on training rows and pad remaining wires with zero rotations.
    final_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)) if scaler == 'MinMax' else StandardScaler()
    Xtr, Xte = final_scaler.fit_transform(Xtr), final_scaler.transform(Xte)
    if scaler != 'MinMax':
        Xtr, Xte = np.clip(Xtr, -np.pi, np.pi), np.clip(Xte, -np.pi, np.pi)
    Xtr = np.pad(Xtr, ((0, 0), (0, qubits - Xtr.shape[1])))
    Xte = np.pad(Xte, ((0, 0), (0, qubits - Xte.shape[1])))
    return PreparedData(Xtr.astype(float), Xte.astype(float), y[train_pos], y[test_pos],
                        data.index.to_numpy()[test_pos], X.columns.tolist(), encoder,
                        {'columns': column_transformer, 'pca': pca, 'angles': final_scaler},
                        len(train_pos), len(test_pos))
