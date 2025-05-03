from dash import dcc, html
from create_figures import create_2d_basis_vectors


def create_graph_section(app) -> html.Section:
    return html.Section(
        [
            dcc.Interval(
                id="animation-interval",
                disabled=True,
                interval=app.interval_ms,
                n_intervals=0,
            ),
            dcc.Store(id="animation-steps", data=[]),
            dcc.Graph(
                id="graph",
                figure=create_2d_basis_vectors(app.BASIS_VECTORS),
            ),
        ]
    )
