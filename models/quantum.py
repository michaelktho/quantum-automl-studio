"""Small, exact state-vector quantum learners. PennyLane runs on CPU."""
from dataclasses import dataclass
import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def _entangle(n_wires):
    for i in range(n_wires - 1):
        qml.CNOT(wires=[i, i + 1])


def _feature_map(x, n_wires, repetitions=1):
    for _ in range(repetitions):
        qml.AngleEmbedding(x, wires=range(n_wires), rotation='Y')
        _entangle(n_wires)
        for i in range(n_wires):
            qml.RZ(x[i], wires=i)


def quantum_kernel_svm(X_train, y_train, X_test, qubits, layers=1, regularisation=1.):
    """Fidelity kernel with a nontrivial data-dependent entangling feature map."""
    dev = qml.device('default.qubit', wires=qubits)

    @qml.qnode(dev)
    def state(x):
        _feature_map(x, qubits, layers)
        return qml.state()

    train_states = np.asarray([state(x) for x in X_train], dtype=complex)
    test_states = np.asarray([state(x) for x in X_test], dtype=complex)
    # |<psi(x)|psi(z)>|^2, calculated once per state rather than repeated QNodes.
    Ktrain = np.clip(np.abs(train_states.conj() @ train_states.T) ** 2, 0, 1)
    Ktest = np.clip(np.abs(test_states.conj() @ train_states.T) ** 2, 0, 1)
    clf = SVC(kernel='precomputed', C=regularisation)
    clf.fit(Ktrain, y_train)
    return clf.predict(Ktest), {'support_vectors': int(len(clf.support_)), 'train_kernel': Ktrain}


@dataclass
class VariationalResult:
    predictions: np.ndarray
    history: list
    circuit: str


def variational_train(X_train, y_train, X_test, task, qubits=2, layers=2,
                      epochs=20, learning_rate=.08, seed=42, progress=None):
    """Train a simple variational circuit with exact expectations and Adam."""
    if task not in ('Classification', 'Regression'):
        raise ValueError('Invalid task.')
    dev = qml.device('default.qubit', wires=qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(qubits))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=qubits)
    rng = np.random.default_rng(seed)
    weights = qml.numpy.array(rng.normal(0, .15, size=shape), requires_grad=True)
    bias = qml.numpy.array(0., requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    X = qml.numpy.array(X_train, requires_grad=False)
    if task == 'Classification':
        targets = qml.numpy.array(np.where(y_train == 1, 1., -1.), requires_grad=False)
        y_centre, y_scale = 0., 1.
    else:
        y_centre = float(np.mean(y_train))
        y_scale = float(np.std(y_train))
        if y_scale < 1e-10:
            y_scale = 1.
        targets = qml.numpy.array((y_train - y_centre) / y_scale, requires_grad=False)

    def cost(w, b):
        predictions = qml.math.stack([circuit(x, w) + b for x in X])
        return qml.numpy.mean((predictions - targets) ** 2)

    history = []
    for i in range(epochs):
        weights, bias = optimizer.step(cost, weights, bias)
        loss = float(cost(weights, bias))
        if not np.isfinite(loss):
            raise ValueError('Quantum optimisation diverged; try a smaller learning rate.')
        history.append(loss)
        if progress is not None:
            progress(i + 1, epochs, loss)
    raw = np.asarray([float(circuit(qml.numpy.array(x, requires_grad=False), weights) + bias)
                      for x in X_test])
    predictions = (raw >= 0).astype(int) if task == 'Classification' else raw * y_scale + y_centre
    diagram = qml.draw(circuit)(qml.numpy.array(X_train[0], requires_grad=False), weights)
    return VariationalResult(predictions, history, diagram)
