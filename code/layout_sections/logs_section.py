from dash import html

def create_logs_section() -> html.Section:
    return html.Section(
        [
            html.Div(
                [
                    html.Label("", id="matrix-list"),
                    html.Label("", id="output-logs"),
                ],
            )
        ]
    )


