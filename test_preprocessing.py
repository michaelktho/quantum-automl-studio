import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from utils.preprocessing import prepare
from utils.evaluation import classical_baseline, evaluate, predictions_frame


def test_classification_pipeline():
    x, y = make_classification(n_samples=90, n_features=5, n_informative=3, n_redundant=0, random_state=3)
    df = pd.DataFrame(x, columns=list('abcde'))
    df['category'] = ['A','B','C'] * 30
    df['target'] = y
    prep = prepare(df, 'target', 'Classification', qubits=3)
    assert prep.X_train.shape[1] == 3
    assert len(set(prep.test_indices).intersection(set(df.index).difference(set(prep.test_indices)))) == 0
    baseline = classical_baseline(prep, 'Classification')
    assert 0 <= evaluate('Classification', prep.y_test, baseline)['Accuracy'] <= 1
    assert len(predictions_frame(prep, baseline, baseline, 'Classification')) == prep.n_test


def test_regression_pipeline():
    x, y = make_regression(n_samples=100, n_features=3, random_state=2)
    df = pd.DataFrame(x, columns=['a','b','c']); df['target'] = y
    prep = prepare(df, 'target', 'Regression', qubits=2)
    pred = classical_baseline(prep, 'Regression')
    assert np.isfinite(evaluate('Regression', prep.y_test, pred)['RMSE'])


def test_invalid_multiclass():
    df = pd.DataFrame({'feature': range(30), 'target': [0,1,2]*10})
    with pytest.raises(ValueError, match='binary'):
        prepare(df, 'target', 'Classification')


def test_unknown_category_in_test_handled():
    df = pd.DataFrame({'x': np.arange(40), 'category':[f'group_{i}' for i in range(40)],
                       'target':[0,1]*20})
    prep = prepare(df, 'target', 'Classification')
    assert np.isfinite(prep.X_test).all()
