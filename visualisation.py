"""Plotly figures for held-out model assessment."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

BLUE = '#4565B4'
TEAL = '#168C83'

def theme(fig):
    fig.update_layout(template='plotly_white', font=dict(family='Arial', color='#233148'),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=30, r=20, t=45, b=35))
    return fig


def comparison(metrics_quantum, metrics_classical):
    names = list(metrics_quantum)
    fig = go.Figure()
    fig.add_bar(name='Quantum', x=names, y=[metrics_quantum[n] for n in names], marker_color=BLUE)
    fig.add_bar(name='Classical', x=names, y=[metrics_classical[n] for n in names], marker_color=TEAL)
    fig.update_layout(barmode='group', title='Held-out model comparison')
    return theme(fig)


def learning_curve(history):
    fig = px.line(x=np.arange(1, len(history)+1), y=history, labels={'x': 'Epoch', 'y': 'Training loss'}, title='Quantum training convergence')
    fig.update_traces(line_color=BLUE)
    return theme(fig)


def regression_scatter(actual, quantum, classical):
    fig = go.Figure()
    fig.add_scatter(x=actual, y=quantum, mode='markers', name='Quantum', marker_color=BLUE)
    fig.add_scatter(x=actual, y=classical, mode='markers', name='Classical', marker_color=TEAL)
    lower = min(np.min(actual), np.min(quantum), np.min(classical))
    upper = max(np.max(actual), np.max(quantum), np.max(classical))
    fig.add_scatter(x=[lower, upper], y=[lower, upper], name='Perfect prediction', mode='lines', line=dict(dash='dash', color='#8895A9'))
    fig.update_layout(title='Actual versus predicted', xaxis_title='Actual', yaxis_title='Predicted')
    return theme(fig)


def confusion(truth, predicted, class_labels):
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale='Blues',
                    x=list(class_labels), y=list(class_labels), labels={'x':'Predicted', 'y':'Actual', 'color':'Count'}, title='Quantum confusion matrix')
    return theme(fig)
