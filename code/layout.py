from dash import dcc, html
from constants import *
import dash_bootstrap_components as dbc
from project_types import *
from layout_sections.vector_section import create_vector_section
from layout_sections.matrix_section import create_matrix_section
from layout_sections.logs_section import create_logs_section
from layout_sections.graph_section import create_graph_section


def create_layout(app) -> html.Main:
    return html.Main(
        [
            html.H1("Matrix Visualizer"),
            create_graph_section(app),
            create_vector_section(app),
            create_matrix_section(),
            create_logs_section(),
        ]
    )
