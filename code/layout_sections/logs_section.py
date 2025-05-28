from dash import html

def create_logs_section() -> html.Section:
    return html.Section(
        id="log-sect",
        children=[
            html.Div(
                [
                    html.Label("", id="output-logs"),
                ],
            )
        ]
    )


