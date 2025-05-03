from dash import dcc, html
from constants import *
import dash_bootstrap_components as dbc
from project_types import *

def create_vector_section(app) -> html.Section:
    return html.Section(
        [
            html.Div(
                [
                    html.H2("Vectors"),
                    dcc.Store(id="vector-store", data={**app.BASIS_VECTORS}),
                    html.Div(
                        [
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
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "new-vector-entry-name",
                                },
                                className="interactable",
                                type="text",
                                placeholder="Name",
                            ),
                            html.Button(
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
                    html.Div(
                        [
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
                            html.Hr(),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "delete-vector-entry-name",
                                },
                                className="interactable",
                                type="text",
                                placeholder="Name",
                            ),
                            html.Button(
                                "Delete Vector",
                                id={
                                    "type": "interactable",
                                    "name": "delete-vector-button",
                                },
                                className="interactable",
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )


