import numpy as np
import pytest
qml = pytest.importorskip('pennylane')
from models.quantum import quantum_kernel_svm, variational_train


def test_quantum_kernel():
    X = np.array([[0.,0.],[.2,.4],[1.,1.],[1.2,1.4]])
    labels = np.array([0,0,1,1])
    predictions, info = quantum_kernel_svm(X, labels, X[:2], 2)
    assert predictions.shape == (2,)
    assert np.allclose(np.diag(info['train_kernel']), 1.)


def test_variational_classifier_and_regressor():
    X = np.array([[.1,.1],[.2,.3],[.7,.8],[.9,1.]])
    yc = np.array([0,0,1,1])
    classification = variational_train(X, yc, X[:2], 'Classification', qubits=2, layers=1, epochs=2)
    assert classification.predictions.shape == (2,)
    assert len(classification.history) == 2
    regression = variational_train(X, np.array([1.,2.,3.,4.]), X[:2], 'Regression', qubits=2, layers=1, epochs=2)
    assert np.isfinite(regression.predictions).all()
