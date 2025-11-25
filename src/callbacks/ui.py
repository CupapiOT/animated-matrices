from dash import callback_context, ALL, html, _callback
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from src.config.constants import LOWER_LETTERS
from src.types import Vectors, MatrixDict
import dash_latex as dl


def register_ui_callbacks(app_instance):
    @app_instance.app.callback(
        Output({"type": "interactable", "name": ALL}, "disabled"),
        [Input("animation-interval", "disabled")],
    )
    def disable_while_animating(
        # Double negative in the name due to the nature of the trait 'disabled'.
        is_not_animating: bool,
    ) -> tuple:
        """Disables and enables any interactable component (e.g.: button,
        entry) based on if an animation is ongoing.
        """
        amount_of_interactables: int = len(callback_context.outputs_list)
        is_animating = not is_not_animating
        return (is_animating,) * amount_of_interactables

    @app_instance.app.callback(
        Output({"type": "interactable", "name": "add-vector-button"}, "children"),
        [
            Input({"type": "interactable", "name": "new-vector-entry-name"}, "value"),
            Input("vector-store", "data"),
            State({"type": "interactable", "name": "add-vector-button"}, "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def change_vector_button_name(
        name_input: str, vectors: Vectors, n_clicks: int
    ) -> str:
        names = [name for name in vectors]
        if name_input in names or LOWER_LETTERS[n_clicks % 26] in names:
            return "Edit Vector"
        else:
            return "Add Vector"

    @app_instance.app.callback(
        Output("matrix-sect__matrix-list", "children", allow_duplicate=True),
        Output("matrix-sect__latest-matrix", "children", allow_duplicate=True),
        [Input("matrix-store", "data")],
        prevent_initial_call=True,
    )
    def update_matrix_list(
        stored_matrices: MatrixDict,
    ) -> tuple[list[html.Li], dl.DashLatex]:
        def smart_format(value):
            return ("%.5f" % value).rstrip("0").rstrip(".")

        new_list: list[str] = []
        for mat_name, ((x1, y1), (x2, y2)) in stored_matrices.items():
            current_matrix = (
                r"""\( %s = \begin{bmatrix} %s & %s \\ %s & %s \end{bmatrix} \)"""
                % (
                    mat_name,
                    smart_format(x1),
                    smart_format(y1),
                    smart_format(x2),
                    smart_format(y2),
                )
            )
            current_matrix = current_matrix.strip()
            new_list.append(current_matrix)
        matrix_list: list[html.Li] = [
            html.Li(
                className="matrix-sect__matrix-list__item",
                children=dl.DashLatex(mat_latex),
            )
            for mat_latex in new_list
        ]
        latest_matrix = new_list[-1] if len(new_list) else ""
        return matrix_list, dl.DashLatex(latest_matrix)

    @app_instance.app.callback(
        Output("log-sect__list", "children", allow_duplicate=True),
        [Input("output-logs", "data")],
        prevent_initial_call=True,
    )
    def update_output_logs_diplay(
        output_logs: list[str],
    ) -> list[html.Li] | _callback.NoUpdate:
        def create_log_span(log: str, repetition_count: int) -> html.Span:
            return html.Span(
                children=[
                    log,
                    html.Span(
                        className="log-repetition-count",
                        children=(
                            f" [×{repetition_count + 1}]"
                            if repetition_count > 0
                            else ""
                        ),
                    ),
                ]
            )

        logs_to_be_displayed: list[html.Span] = []
        try:
            previous_log = output_logs[0]
        except IndexError:
            print(
                "Warning: Output logs was updated but there are no logs to display. "
                "This is usually harmless."
            )
            raise PreventUpdate
        stack_repeat_count = 0
        for log in output_logs[1:]:
            if log == previous_log:
                stack_repeat_count += 1
                continue
            logs_to_be_displayed.append(
                create_log_span(previous_log, stack_repeat_count)
            )
            previous_log = log
            stack_repeat_count = 0
        if previous_log is not None:
            logs_to_be_displayed.append(
                create_log_span(previous_log, stack_repeat_count)
            )

        return [
            html.Li(className="log-sect__list__log", children=log)
            for log in logs_to_be_displayed
        ]
