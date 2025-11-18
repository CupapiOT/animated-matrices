from dash import html, dcc
import dash_bootstrap_components as dbc


def create_logs_section() -> html.Section:
    return html.Section(
        id="log-sect",
        children=[
            dcc.Store(id="output-logs", data=[]),
            dbc.Accordion(
                id="log-sect__display",
                children=[
                    dbc.AccordionItem(
                        children=[
                            html.Ol(
                                id="log-sect__list",
                                children="No logs yet.",
                            ),
                        ],
                        title="Logs",
                    )
                ],
                start_collapsed=True,
            ),
        ],
    )
