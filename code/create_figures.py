from plotly.graph_objects import Figure, Scatter
from project_types import Vectors

__all__ = ['create_2d_basis_vectors', 'create_figure']


def create_2d_basis_vectors(
        basis_vectors: Vectors
        ):
    fig = Figure()

    fig.add_trace(Scatter(
        x=[0, basis_vectors['i-hat'].coords[0]],
        y=[0, basis_vectors['i-hat'].coords[1]],
        mode='lines+markers+text',
        text=['', 'i-hat'],
        textposition='top center',
        line=dict(color='green'),
        name='i-hat'
    ))

    fig.add_trace(Scatter(
        x=[0, basis_vectors['j-hat'].coords[0]],
        y=[0, basis_vectors['j-hat'].coords[1]],
        mode='lines+markers+text',
        text=['', 'j-hat'],
        textposition='top center',
        line=dict(color='red'),
        name='j-hat'
    ))

    _update_fig_layout(fig)

    return fig


def create_figure(vectors: Vectors) -> Figure:
    fig = Figure()

    for name, vector in vectors.items():
        fig.add_trace(Scatter(
            x=[0, vector.coords[0]],
            y=[0, vector.coords[1]],
            mode='lines+markers+text',
            text=['', name],
            textposition='top center',
            line=dict(color=vector.color),
            name=name
        ))

    _update_fig_layout(fig)

    return fig


def _update_fig_layout(fig) -> None:
    fig.update_layout(
        title='Graph',
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5], scaleanchor='x'),
        showlegend=True
    )
