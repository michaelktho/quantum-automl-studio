"""Unambiguous held-out evaluation and downloadable predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.svm import SVC, SVR


def classical_baseline(prepared, task, regularisation=1.):
    model = SVC(C=regularisation, kernel='rbf') if task == 'Classification' else SVR(C=regularisation)
    model.fit(prepared.X_train, prepared.y_train)
    return model.predict(prepared.X_test)


def evaluate(task, truth, predictions):
    if task == 'Classification':
        return {'Accuracy': accuracy_score(truth, predictions),
                'Precision': precision_score(truth, predictions, zero_division=0),
                'Recall': recall_score(truth, predictions, zero_division=0),
                'F1': f1_score(truth, predictions, zero_division=0)}
    mse = mean_squared_error(truth, predictions)
    return {'MAE': mean_absolute_error(truth, predictions), 'MSE': mse,
            'RMSE': np.sqrt(mse), 'R²': r2_score(truth, predictions) if len(truth) >= 2 else np.nan}


def predictions_frame(prepared, quantum, classical, task):
    actual, q, c = prepared.y_test, quantum, classical
    if task == 'Classification':
        enc = prepared.label_encoder
        actual, q, c = enc.inverse_transform(np.asarray(actual, dtype=int)), enc.inverse_transform(np.asarray(q, dtype=int)), enc.inverse_transform(np.asarray(c, dtype=int))
    return pd.DataFrame({'source_row_index': prepared.test_indices, 'actual': actual,
                         'quantum_prediction': q, 'classical_prediction': c})
