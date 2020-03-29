import plotly.graph_objects as go
import numpy as np


class Plot2D:
    def __init__(self, x_label='Displacement', y_label=r'Load factor (λ)', title='Load-displacement diagram'):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.invert_xaxis = True
        self.data = []

    def add_load_displacement_curve(self, model, dof, label=None, show_iterations=False):
        _add_load_displacement_curve(self.data, model, dof, label)

        if show_iterations:
            _plot_load_displacement_iterations(self.data, model, dof, label)

    def add_det_k_curve(self, model, dof, label=None):
        _plot_det_k_curve(self.data, model, dof, label)

    def add_history_curve(self, model, fun, label, show_iterations=False):
        _plot_history_curve(self.data, model, fun, label, not show_iterations)

    def add_custom_curve(self, x, y, label, linewidth=1.0):
        self.data.append(go.Scatter(
            name=label,
            mode='lines+markers',
            x=x,
            y=y,
            line=dict(
                width=linewidth,
            ),
        ))

    def show(self, height=500):
        fig = go.Figure(
            data=self.data,
            layout=go.Layout(
                title=self.title,
                xaxis=dict(
                    autorange='reversed',
                    title_text=self.x_label,
                ),
                yaxis=dict(
                    title_text=self.y_label,
                ),
            ),
        )
        fig.show()


def show_load_displacement_curve(model, dof):
    dof_type, node_id = dof

    plot = Plot2D()
    model.add_load_displacement_curve(model, dof)

    plot.show()


def _add_load_displacement_curve(data, model, dof, label):
    history = model.get_model_history()

    x = np.zeros(len(history))
    y = np.zeros(len(history))

    for i, model in enumerate(history):
        x[i] = model[dof].delta
        y[i] = model.load_factor

    node_id, dof_type = dof

    if label is None:
        label = f'λ : {dof_type} at node {node_id}'

    data.append(go.Scatter(
        name=label,
        mode='lines+markers',
        x=x,
        y=y,
    ))


def _plot_load_displacement_iterations(data, model, dof, label):
    history = model.get_model_history(skip_iterations=False)

    x = np.zeros(len(history))
    y = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x[i] = model[dof].delta
        y[i] = model.load_factor

    if label is None:
        label = f'λ : {dof_type} at node {node_id} (iter)'
    else:
        label += ' (iter)'

    data.append(go.Scatter(
        name=label,
        mode='lines+markers',
        x=x,
        y=y,
        line=dict(
            width=0.75,
        ),
        marker=dict(
            size=2.0,
        ),
    ))


def _plot_det_k_curve(data, model, dof, label):
    history = model.get_model_history()

    x = np.zeros(len(history))
    y = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x[i] = model[dof].delta
        y[i] = model.det_k

    if label is None:
        label = 'det(K) : {} at node {}'.format(dof_type, node_id)

    data.append(go.Scatter(
        name=label,
        mode='lines+markers',
        x=x,
        y=y,
    ))


def _plot_history_curve(data, model, fun, label, skip_iterations):
    history = model.get_model_history(skip_iterations)

    x = np.zeros(len(history))
    y = np.zeros(len(history))

    for i, model in enumerate(history):
        x[i], y[i] = fun(model)

    data.append(go.Scatter(
        name=label,
        mode='lines+markers',
        x=x,
        y=y,
    ))
