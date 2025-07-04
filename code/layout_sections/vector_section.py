from dash import dcc, html
from constants import *
import dash_bootstrap_components as dbc
from project_types import *
from layout_sections.layout_components import matrix_component


def create_vector_section(app) -> html.Section:
    return html.Section(
        id="vector-sect",
        children=[
            dcc.Store(id="vector-store", data={**app.BASIS_VECTORS}),
            matrix_component.input_matrix(
                id="vector-sect__coordinates",
                inputs=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "vector-entry-1",
                        },
                        className="interactable",
                        type="number",
                        placeholder="x",
                    ),
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "vector-entry-2",
                        },
                        className="interactable",
                        type="number",
                        placeholder="y",
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                id="vector-sect__add-vector",
                className="entry-input-pair",
                children=[
                    dcc.Dropdown(
                        id={
                            "type": "interactable",
                            "name": "vector-entry-color",
                        },
                        className="interactable",
                        options=[
                            {
                                "label": color.capitalize(),
                                "value": color,
                            }
                            for color in COLORS
                        ],
                        value="black",
                    ),
                    html.Div(
                        className="entry-input-pair",
                        children=[
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "new-vector-entry-name",
                                },
                                type="text",
                                placeholder="Name",
                            ),
                            dbc.Button(
                                children="Add Vector",
                                id={
                                    "type": "interactable",
                                    "name": "add-vector-button",
                                },
                                className="interactable",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                className="entry-input-pair",
                children=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "delete-vector-entry-name",
                        },
                        className="interactable",
                        type="text",
                        placeholder="Name",
                    ),
                    dbc.Button(
                        "Delete Vector",
                        id={
                            "type": "interactable",
                            "name": "delete-vector-button",
                        },
                        className="interactable",
                    ),
                ],
            ),
        ],
    )
