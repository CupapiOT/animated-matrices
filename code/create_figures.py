from plotly.graph_objects import Figure, Scatter
from types import Vectors

__all__ = ['create_2d_basis_vectors', 'create_figure']


def create_2d_basis_vectors(
        basis_vectors: Vectors
        ):
    fig = Figure()

    fig.add_trace(Scatter(
        x=[0, basis_vectors['i-hat'][0][0]],
        y=[0, basis_vectors['i-hat'][0][1]],
        mode='lines+markers+text',
        text=['', 'i-hat'],
        textposition='top center',
        line=dict(color='green'),
        name='i-hat'
    ))

    fig.add_trace(Scatter(
        x=[0, basis_vectors['j-hat'][0][0]],
        y=[0, basis_vectors['j-hat'][0][1]],
        mode='lines+markers+text',
        text=['', 'j-hat'],
        textposition='top center',
        line=dict(color='red'),
        name='j-hat'
    ))

    fig = update_fig_layout(fig)
    
    return fig


def create_figure(vectors: Vectors) -> Figure:
    fig = Figure()

    for name, ((x, y), color) in vectors.items():
        fig.add_trace(Scatter(
            x=[0, x],
            y=[0, y],
            mode='lines+markers+text',
            text=['', name],
            textposition='top center',
            line=dict(color=color),
            name=name
        ))

    fig = update_fig_layout(fig)

    return fig


def update_fig_layout(fig) -> Figure:
    fig.update_layout(
        title='Graph',
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5], scaleanchor='x'),
        showlegend=True
    )
    return fig
