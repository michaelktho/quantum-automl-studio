"""Quantum AutoML Studio | Streamlit entrypoint."""
import hashlib
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification, make_regression

from utils.preprocessing import read_csv, prepare
from utils.evaluation import classical_baseline, evaluate, predictions_frame
from utils.visualisation import comparison, learning_curve, regression_scatter, confusion

st.set_page_config(page_title='Quantum AutoML Studio', page_icon='⚛️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
:root {color-scheme:light;} .stApp {background:#f8fafc;color:#203049;}
.block-container {max-width:1240px;padding-top:2.2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;color:#203049!important;}
h1 {font-weight:730!important;} [data-testid="stMetric"] {background:#fff;border:1px solid #e4eaf1;border-radius:12px;padding:16px;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e8edf4;}
.stButton button[kind="primary"] {background:#4565b4;border:0;border-radius:9px;}
div[data-testid="stAlert"] {border-radius:10px;}
.tagline {color:#64748b;font-size:1rem;margin-top:-.45rem;margin-bottom:1.7rem;}
.step {font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#4565b4;font-weight:700;}
.footer {color:#7c899c;font-size:.78rem;border-top:1px solid #e5eaf1;margin-top:2rem;padding-top:1rem;}
</style>''', unsafe_allow_html=True)


def demo_dataset(task):
    if task == 'Classification':
        X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
    else:
        X, y = make_regression(n_samples=120, n_features=4, n_informative=3,
                               noise=12, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target'] = y
    return df


with st.sidebar:
    st.markdown('### ⚛️ Quantum AutoML')
    st.caption('Scientific Machine Learning Studio')
    st.divider()
    st.markdown('**Experiment setup**')
    task = st.radio('Learning task', ['Classification', 'Regression'])
    upload = st.file_uploader('Upload your CSV', type=['csv'], help='CSV up to 5 MB. Do not upload sensitive or confidential data.')
    if upload and upload.size > 5 * 1024 * 1024:
        st.error('The CSV exceeds the 5 MB limit.')
        st.stop()
    use_demo = st.checkbox('Use an example dataset', value=upload is None)
    st.caption('Uploaded files are handled in the active app session; no permanent dataset storage is implemented.')
    st.divider()
    st.caption('Local quantum state-vector simulation · No quantum hardware required')

st.markdown('<div class="step">Quantum computing / Applied machine learning</div>', unsafe_allow_html=True)
st.title('Quantum AutoML Studio')
st.markdown('<div class="tagline">Upload a dataset. Train a quantum model. Evaluate it alongside a classical baseline.</div>', unsafe_allow_html=True)

if upload is None and not use_demo:
    st.info('Upload a CSV or enable the example dataset to begin.')
    st.stop()
try:
    if upload is not None:
        dataset_bytes = upload.getvalue()
        df = read_csv(io.BytesIO(dataset_bytes))
        dataset_id = hashlib.sha256(dataset_bytes).hexdigest()
    else:
        df = demo_dataset(task)
        dataset_id = f'demo-{task}'
except Exception as exc:
    st.error(f'Could not read the dataset: {exc}')
    st.stop()

tab_data, tab_config, tab_results = st.tabs(['01  Dataset', '02  Configure & train', '03  Results'])
with tab_data:
    st.subheader('Your dataset')
    a,b,c = st.columns(3)
    a.metric('Observations', f'{len(df):,}')
    b.metric('Columns', f'{len(df.columns):,}')
    c.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.caption('Preview shows the first 30 rows. The training cap may sample observations reproducibly.')
    with st.expander('Column overview'):
        overview = pd.DataFrame({'Type':df.dtypes.astype(str), 'Missing':df.isna().sum(), 'Unique':df.nunique(dropna=True)})
        st.dataframe(overview, use_container_width=True)

with tab_config:
    st.subheader('Experiment configuration')
    left, right = st.columns([1, 1], gap='large')
    with left:
        target = st.selectbox('Target column', options=list(df.columns), index=len(df.columns)-1,
                              help='The target will never be included among the input features.')
        model_options = ['Variational quantum classifier', 'Quantum kernel SVM'] if task == 'Classification' else ['Variational quantum regressor']
        model = st.selectbox('Quantum model', model_options)
        test_pct = st.slider('Test set (%)', min_value=10, max_value=40, value=20, step=5)
        seed = st.number_input('Random seed', min_value=0, max_value=999999, value=42)
        row_limit = st.slider('Maximum observations to sample', 30, 500, 120, 10,
                              help='A deterministic sample is drawn before splitting if there are more rows than this.')
    with right:
        qubits = st.slider('Number of qubits', 2, 4, 2)
        layers = st.slider('Circuit layers / feature-map repetitions', 1, 3, 2)
        scaling = st.selectbox('Angle scaling', ['Standard', 'MinMax'])
        regularisation = st.slider('SVM regularisation (C)', .1, 10., 1., .1)
        if not model.startswith('Quantum kernel'):
            epochs = st.slider('Optimisation epochs', 5, 50, 15, 5)
            learning_rate = st.slider('Learning rate', .01, .20, .08, .01)
        else:
            epochs, learning_rate = None, None
    st.caption('Missing values are imputed, categorical features one-hot encoded, and high-dimensional inputs reduced with PCA. All transformations are fitted on the training partition only. Binary classification only in v1.')
    experiment = {'dataset': dataset_id, 'task': task, 'target': target, 'model': model,
                  'test_pct': test_pct, 'seed': int(seed), 'rows': row_limit, 'qubits': qubits,
                  'layers': layers, 'scaling': scaling, 'C': regularisation,
                  'epochs': epochs, 'lr': learning_rate}
    experiment_id = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    train = st.button('⚛️  Train and evaluate models', type='primary', use_container_width=True)
    if train:
        st.session_state.pop('experiment_result', None)
        try:
            with st.status('Preparing your experiment…', expanded=True) as status:
                prepared = prepare(df, target, task, qubits=qubits,
                                   test_size=test_pct/100, seed=int(seed),
                                   scaler=scaling, sample_limit=row_limit)
                st.write(f'Prepared {prepared.n_train} training rows and {prepared.n_test} test rows.')
                if model == 'Quantum kernel SVM' and prepared.n_train > 180:
                    raise ValueError('Quantum kernel SVM is restricted to 180 training rows due to quadratic memory and compute costs. Reduce maximum observations.')
                start = time.perf_counter()
                if model == 'Quantum kernel SVM':
                    from models.quantum import quantum_kernel_svm
                    qp, details = quantum_kernel_svm(prepared.X_train, prepared.y_train,
                                                       prepared.X_test, qubits, layers, regularisation)
                    history, circuit_text = [], None
                else:
                    from models.quantum import variational_train
                    progress = st.progress(0, text='Optimising quantum circuit')
                    def update(current, total, loss):
                        progress.progress(current / total, text=f'Epoch {current}/{total} · loss {loss:.4f}')
                    result = variational_train(prepared.X_train, prepared.y_train, prepared.X_test,
                                               task, qubits, layers, epochs, learning_rate, int(seed), update)
                    qp, history, circuit_text = result.predictions, result.history, result.circuit
                    progress.empty()
                quantum_seconds = time.perf_counter() - start
                cp = classical_baseline(prepared, task, regularisation)
                qmetrics, cmetrics = evaluate(task, prepared.y_test, qp), evaluate(task, prepared.y_test, cp)
                pred_df = predictions_frame(prepared, qp, cp, task)
                st.session_state['experiment_result'] = {'id': experiment_id, 'task': task,
                    'qmetrics': qmetrics, 'cmetrics': cmetrics, 'pred_df': pred_df,
                    'history': history, 'circuit': circuit_text, 'seconds': quantum_seconds,
                    'settings': experiment, 'n_train': prepared.n_train, 'n_test': prepared.n_test,
                    'actual_numeric': prepared.y_test, 'quantum_numeric': qp,
                    'classical_numeric': cp,
                    'class_labels': prepared.label_encoder.classes_.tolist() if prepared.label_encoder is not None else None}
                status.update(label='Training and evaluation completed', state='complete', expanded=False)
        except Exception as exc:
            st.error(f'The experiment could not complete: {exc}')
            st.info('Try fewer rows, a simpler model, or a different target column.')

with tab_results:
    result = st.session_state.get('experiment_result')
    if result is None:
        st.info('Configure and train a model to populate this dashboard.')
    elif result['id'] != experiment_id:
        st.warning('Your configuration or dataset has changed. Train again to view results for these settings.')
    else:
        st.subheader('Held-out test results')
        st.caption(f"{result['n_train']} train · {result['n_test']} test · Quantum computation {result['seconds']:.2f} s · Same split for both models")
        metrics = result['qmetrics']
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f'{value:.3f}' if np.isfinite(value) else 'N/A')
        st.plotly_chart(comparison(result['qmetrics'], result['cmetrics']), use_container_width=True)
        c1, c2 = st.columns(2, gap='large')
        with c1:
            if task == 'Classification':
                st.plotly_chart(confusion(result['actual_numeric'], result['quantum_numeric'], result['class_labels']), use_container_width=True)
            else:
                st.plotly_chart(regression_scatter(result['actual_numeric'], result['quantum_numeric'], result['classical_numeric']), use_container_width=True)
        with c2:
            if result['history']:
                st.plotly_chart(learning_curve(result['history']), use_container_width=True)
            else:
                st.info('A kernel SVM has no epoch-by-epoch training loss.')
            st.markdown('**Experiment metadata**')
            st.json(result['settings'], expanded=False)
        with st.expander('Quantum circuit representation'):
            if result['circuit']:
                st.code(result['circuit'], language='text')
            else:
                st.caption('The quantum kernel evaluates fidelity between feature-map states. Circuit drawing is available for variational models.')
        st.markdown('#### Test-set predictions')
        st.dataframe(result['pred_df'].head(100), use_container_width=True, hide_index=True)
        st.download_button('Download test predictions (CSV)', result['pred_df'].to_csv(index=False),
                           'quantum_test_predictions.csv', mime='text/csv', use_container_width=True)
        report = {'configuration': result['settings'], 'train_rows': result['n_train'],
                  'test_rows': result['n_test'], 'quantum_metrics': result['qmetrics'],
                  'classical_metrics': result['cmetrics'], 'quantum_training_seconds': result['seconds']}
        st.download_button('Download experiment report (JSON)', json.dumps(report, indent=2),
                           'quantum_experiment.json', mime='application/json', use_container_width=True)

st.markdown('<div class="footer">Research demonstration · Local state-vector simulation · Test scores are not proof of quantum advantage · Please avoid uploading confidential data.</div>', unsafe_allow_html=True)
