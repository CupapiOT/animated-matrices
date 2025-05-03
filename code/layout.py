from dash import dcc, html
from constants import *
from create_figures import create_2d_basis_vectors
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
                                style={
                                    "marginBottom": "10px",
                                    "width": "45%",
                                    "marginRight": "5%",
                                },
                                placeholder="x",
                            ),
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "vector-entry-2",
                                },
                                className="interactable",
                                type="number",
                                style={
                                    "marginBottom": "10px",
                                    "width": "45%",
                                },
                                size="2",
                                placeholder="y",
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
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
                                style={
                                    "marginRight": "5%",
                                    "marginBottom": "5%",
                                    "width": "20%",
                                },
                                size="2",
                                placeholder="Name",
                            ),
                            html.Button(
                                children="Add Vector",
                                id={
                                    "type": "interactable",
                                    "name": "add-vector-button",
                                },
                                className="interactable",
                                style={
                                    "marginBottom": "5%",
                                    "width": "70%",
                                },
                                n_clicks=0,
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
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
                            html.Hr(
                                style={
                                    "width": "100%",
                                    "marginBottom": "15px",
                                    "marginTop": "15px",
                                }
                            ),
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
                                style={
                                    "marginRight": "5%",
                                    "marginBottom": "5%",
                                    "width": "20%",
                                },
                                size="2",
                                placeholder="Name",
                            ),
                            html.Button(
                                "Delete Vector",
                                id={
                                    "type": "interactable",
                                    "name": "delete-vector-button",
                                },
                                className="interactable",
                                style={
                                    "marginBottom": "5%",
                                    "width": "70%",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "marginLeft": "20px",
                    "width": "200px",
                },
            ),
        ]
    )


def create_matrix_section() -> html.Section:
    return html.Section(
        [
            html.H2("Matrices"),
            dcc.Store(id="matrix-store", data={}),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "matrix-entry-1",
                                },
                                className="interactable",
                                type="number",
                                style={
                                    "marginBottom": "10px",
                                    "marginRight": "10px",
                                    "width": "80px",
                                },
                                size="2",
                                placeholder="a",
                            ),
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "matrix-entry-2",
                                },
                                className="interactable",
                                type="number",
                                style={
                                    "marginBottom": "10px",
                                    "width": "80px",
                                },
                                size="2",
                                placeholder="b",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "matrix-entry-3",
                                },
                                className="interactable",
                                type="number",
                                style={
                                    "marginBottom": "10px",
                                    "marginRight": "10px",
                                    "width": "80px",
                                },
                                size="2",
                                placeholder="c",
                            ),
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "matrix-entry-4",
                                },
                                className="interactable",
                                type="number",
                                style={
                                    "marginBottom": "10px",
                                    "width": "80px",
                                },
                                size="2",
                                placeholder="d",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "new-matrix-entry-name",
                                },
                                className="interactable",
                                type="text",
                                style={
                                    "width": "20%",
                                    "marginRight": "5%",
                                },
                                size="2",
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
                                style={"width": "70%"},
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),
                    html.Hr(
                        style={
                            "marginBottom": "15px",
                            "marginTop": "15px",
                            "width": "100%",
                        }
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id={
                                    "type": "interactable",
                                    "name": "inverse-matrix-entry-name",
                                },
                                className="interactable",
                                type="text",
                                style={
                                    "width": "20%",
                                    "marginRight": "5%",
                                },
                                size="2",
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
                                style={"width": "70%"},
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),
                    html.Hr(
                        style={
                            "marginBottom": "15px",
                            "marginTop": "15px",
                            "width": "100%",
                        }
                    ),
                    html.Div(
                        [
                            dcc.Store(id="previous-vector-store", data=[]),
                            html.Button(
                                "Undo Last Matrix",
                                id={
                                    "type": "interactable",
                                    "name": "undo-matrix-button",
                                },
                                className="interactable",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "marginBottom": "5%",
                                },
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
                                style={"width": "100%"},
                            ),
                            html.Hr(
                                style={
                                    "marginBottom": "15px",
                                    "marginTop": "15px",
                                    "width": "100%",
                                }
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id={
                                            "type": "interactable",
                                            "name": "repeat-matrix-entry-name",
                                        },
                                        className="interactable",
                                        type="text",
                                        style={
                                            "width": "20%",
                                            "marginRight": "5%",
                                        },
                                        size="2",
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
                                        style={"width": "70%"},
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "marginLeft": "20px",
            "width": "200px",
        },
    )


def create_logs_section() -> html.Section:
    return html.Section(
        [
            html.Div(
                [
                    html.Label("List of Matrices", style={"marginBottom": "10px"}),
                    html.Label("", id="matrix-list"),
                    *([html.Br()] * 4),
                    html.Label("Recent Logs:", style={"marginBottom": "10px"}),
                    html.Label("", id="output-logs"),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "500px",
                },
            )
        ]
    )


def create_layout(app) -> html.Div:
    return html.Div(
        [
            html.H1("Matrix Visualizer"),
            html.Div(
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
                        style={
                            "height": "1000%",
                            "width": "70%",
                            "display": "inline-block",
                        },
                    ),
                    create_vector_section(app),
                    create_matrix_section(),
                ],
                style={"display": "flex", "flexDirection": "row"},
            ),
            create_logs_section()
        ]
    )
