from dash import dcc, html
from constants import *
import dash_bootstrap_components as dbc
from project_types import *
from layout_sections.vector_section import create_vector_section
from layout_sections.matrix_section import create_matrix_section
from layout_sections.logs_section import create_logs_section
from layout_sections.graph_section import create_graph_section

def create_layout(app) -> html.Main:
    vector_tab = dbc.Card(dbc.CardBody(create_vector_section(app)))
    matrix_tab = dbc.Card(dbc.CardBody(create_matrix_section()))
    logs_tab = dbc.Card(dbc.CardBody(create_logs_section()))
    return html.Main(
        [
            html.H1("2D Matrix Visualizer", className="visually-hidden"),
            create_graph_section(app),
            dbc.Tabs(
                [
                    dbc.Tab(vector_tab, label="Vectors"),
                    dbc.Tab(matrix_tab, label="Matrices"),
                    dbc.Tab(logs_tab, label="Logs"),
                ],
                id="section-tabs"
            ),
        ]
    )
