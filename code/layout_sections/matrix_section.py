from dash import dcc, html
from constants import *
import dash_bootstrap_components as dbc
from project_types import *
from layout_sections.layout_components import matrix_component


def create_matrix_section() -> html.Section:
    return html.Section(
        id="matrix-sect",
        children=[
            dcc.Store(id="matrix-store", data={}),
            matrix_component.input_matrix(
                id="matrix-sect__coordinates",
                inputs=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "matrix-entry-1",
                        },
                        className="interactable",
                        type="number",
                        placeholder="a",
                    ),
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "matrix-entry-2",
                        },
                        className="interactable",
                        type="number",
                        placeholder="b",
                    ),
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "matrix-entry-3",
                        },
                        className="interactable",
                        type="number",
                        placeholder="c",
                    ),
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "matrix-entry-4",
                        },
                        className="interactable",
                        type="number",
                        placeholder="d",
                    ),
                ],
                ab_id="matrix-sect__a-b",
                cd_id="matrix-sect__c-d",
            ),
            html.Div(
                id="matrix-sect__add-submit",
                children=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "new-matrix-entry-name",
                        },
                        className="interactable",
                        type="text",
                        placeholder="Name",
                    ),
                    html.Button(
                        "Add Matrix",
                        id={
                            "type": "interactable",
                            "name": "add-matrix-button",
                        },
                        className="interactable",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                id="matrix-sect__inverse",
                children=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "inverse-matrix-entry-name",
                        },
                        className="interactable",
                        type="text",
                        placeholder="Name",
                    ),
                    html.Button(
                        "Apply Inverse",
                        id={
                            "type": "interactable",
                            "name": "inverse-matrix-button",
                        },
                        className="interactable",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                id="matrix-sect__undo-redo",
                children=[
                    dcc.Store(id="previous-vector-store", data=[]),
                    html.Button(
                        "Undo Last Matrix",
                        id={
                            "type": "interactable",
                            "name": "undo-matrix-button",
                        },
                        className="interactable",
                        n_clicks=0,
                    ),
                    dcc.Store(id="undone-matrices-store", data={}),
                    html.Button(
                        "Redo Last Matrix",
                        id={
                            "type": "interactable",
                            "name": "redo-matrix-button",
                        },
                        className="interactable",
                        n_clicks=0,
                    ),
                    html.Hr(),
                ],
            ),
            html.Div(
                id="matrix-sect__repeat",
                children=[
                    dcc.Input(
                        id={
                            "type": "interactable",
                            "name": "repeat-matrix-entry-name",
                        },
                        className="interactable",
                        type="text",
                        placeholder="Name",
                    ),
                    html.Button(
                        "Repeat Matrix",
                        id={
                            "type": "interactable",
                            "name": "repeat-matrix-button",
                        },
                        className="interactable",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Hr(),
            dbc.Tabs(
                [
                    dbc.Tab(
                        html.Div(
                            "No matrices right now.",
                            id="matrix-sect__matrix-list",
                            className="matrix-section__logs",
                        ),
                        label="All Matrices",
                        tab_id="matrix-sect__all-matrices",
                    ),
                    dbc.Tab(
                        html.Div(
                            "No matrices right now.",
                            id="matrix-sect__latest-matrix",
                            className="matrix-section__logs",
                        ),
                        label="Latest Matrix",
                        tab_id="matrix-sect__latest-matrix",
                    ),
                ]
            ),
        ],
    )
