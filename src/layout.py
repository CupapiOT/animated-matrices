from dash import html
import dash_bootstrap_components as dbc
from src.config.constants import *
from src.types import *
from src.tabs.vector import create_vector_section
from src.tabs.matrix import create_matrix_section
from src.components.logs_panel import create_logs_section
from src.components.graph import create_graph_section


def create_layout(app) -> html.Main:
    vector_tab = dbc.Card(dbc.CardBody(create_vector_section(app)))
    matrix_tab = dbc.Card(dbc.CardBody(create_matrix_section()))
    return html.Main(
        [
            html.H1("2D Matrix Visualizer", className="visually-hidden"),
            create_graph_section(app),
            dbc.Tabs(
                [
                    dbc.Tab(matrix_tab, label="Matrices", tab_id="matrix-section-tab"),
                    dbc.Tab(vector_tab, label="Vectors", tab_id="vector-section-tab"),
                ],
                id="section-tabs",
                active_tab="matrix-section-tab",
            ),
            create_logs_section(),
        ],
    )
