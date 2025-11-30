import numpy as np
from dash import no_update
from dash.dependencies import Input, Output, State
from src.types import Matrix, MatrixDict, Vectors, Number
from src.utils.matrix import (
    apply_matrix_to_vectors,
    generate_new_matrix_name,
    generate_duplicate_matrix_name,
    generate_inverse_matrix_name,
    is_inverse_matrix,
    safe_inverse,
)
from src.utils.general import set_nonetype_to_zero


def _handle_newly_added_vectors(
    stored_vectors: Vectors,
    previous_vectors: list[Vectors],
    inverse_matrix: Matrix | None,
) -> tuple[list[Vectors], str | None]:
    """Only used within the undo_matrix method."""
    if len(stored_vectors) <= len(previous_vectors[-1]):
        return previous_vectors, None

    new_output_log = None
    new_previous_vectors = previous_vectors.copy()
    new_keys = set(stored_vectors) - (set(new_previous_vectors[-1]))
    new_vector_dict = {key: stored_vectors[key] for key in new_keys}
    if inverse_matrix is not None:
        inverted_new_vectors = apply_matrix_to_vectors(inverse_matrix, new_vector_dict)
        new_previous_vectors[-1].update(inverted_new_vectors)
    else:
        new_previous_vectors[-1].update(new_vector_dict)
        new_output_log = (
            f"The coordinates of newly added vector(s) {list(new_keys)} "
            f"were not updated due to how the previous matrix was unable "
            f"to be inverted. "
        )

    return new_previous_vectors, new_output_log


def register_matrix_callbacks(app_instance):
    @app_instance.app.callback(
        Output("matrix-store", "data", allow_duplicate=True),
        # Output('graph', 'figure', allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        Output("undone-matrices-store", "data", allow_duplicate=True),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("animation-steps", "data", allow_duplicate=True),
        [
            Input({"type": "interactable", "name": "add-matrix-button"}, "n_clicks"),
            State({"type": "interactable", "name": "matrix-entry-1"}, "value"),
            State({"type": "interactable", "name": "matrix-entry-2"}, "value"),
            State({"type": "interactable", "name": "matrix-entry-3"}, "value"),
            State({"type": "interactable", "name": "matrix-entry-4"}, "value"),
        ],
        [
            State("matrix-store", "data"),
            State("vector-store", "data"),
            State("previous-vector-store", "data"),
            State({"type": "interactable", "name": "new-matrix-entry-name"}, "value"),
        ],
        [State("animation-steps", "data")],
        prevent_initial_call=True,
    )
    def add_matrix(
        _: int,
        a: Number,
        b: Number,
        c: Number,
        d: Number,
        stored_matrices: MatrixDict,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        name: str,
        animation_steps: list[Matrix],
    ) -> tuple:
        a, b, c, d = set_nonetype_to_zero(a, b, c, d)

        matrix = np.array([[a, b], [c, d]])
        matrix_name = (
            name if name else generate_new_matrix_name(list(stored_matrices.keys()))
        )

        stored_matrices[matrix_name] = matrix.tolist()

        previous_vectors.append(stored_vectors.copy())

        most_recent_matrix = np.array(list(stored_matrices.values())[-1])
        new_vectors = apply_matrix_to_vectors(most_recent_matrix, stored_vectors)

        new_steps = app_instance.update_animations(
            animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix
        )

        return (
            stored_matrices,
            # create_figure(restored_vectors),
            new_vectors,
            previous_vectors,
            {},  # Empty undone matrices.
            False,
            new_steps,
        )

    @app_instance.app.callback(
        Output("matrix-store", "data", allow_duplicate=True),
        # Output('graph', 'figure', allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        Output("undone-matrices-store", "data", allow_duplicate=True),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("animation-steps", "data", allow_duplicate=True),
        Output("output-logs", "data", allow_duplicate=True),
        [
            Input(
                {"type": "interactable", "name": "inverse-matrix-button"},
                "n_clicks",
            ),
            State(
                {"type": "interactable", "name": "inverse-matrix-entry-name"},
                "value",
            ),
        ],
        [
            State("matrix-store", "data"),
            State("vector-store", "data"),
            State("previous-vector-store", "data"),
            State("output-logs", "data"),
        ],
        [State("animation-steps", "data")],
        prevent_initial_call=True,
    )
    def apply_inverse(
        _,
        matrix_to_invert: str | None,
        stored_matrices: MatrixDict,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        output_logs: list[str],
        animation_steps: list[Matrix],
    ) -> tuple:
        def _validate_input(
            name: str | None,
            stored_matrices_: MatrixDict | tuple | list,
            output_logs_: list[str],
        ) -> tuple[bool, list[str]]:
            if not stored_matrices_:
                output_logs_.append("Apply Inverse: No matrices exist. ")
                return False, output_logs_
            if name and (name not in stored_matrices_):
                output_logs_.append(
                    f"Apply Inverse: Matrix '{matrix_to_invert}' does not exist."
                )
                return False, output_logs_

            return True, output_logs_

        def _get_last_matrix_name(stored_matrices_: MatrixDict) -> str:
            non_inverse_matrices = [
                matrix_name
                for matrix_name in stored_matrices_.keys()
                if not is_inverse_matrix(matrix_name)
            ]
            return (
                non_inverse_matrices[-1]
                if (non_inverse_matrices)
                else tuple(stored_matrices_)[-1]
            )

        valid, new_output_logs = _validate_input(
            name=matrix_to_invert,
            stored_matrices_=stored_matrices,
            output_logs_=output_logs,
        )
        if not valid:
            return (no_update,) * 6 + (new_output_logs,)

        name = (
            matrix_to_invert
            if matrix_to_invert
            else (_get_last_matrix_name(stored_matrices))
        )

        selected_matrix = np.array(stored_matrices[name])
        inverted_matrix = safe_inverse(selected_matrix)
        if inverted_matrix is None:
            new_output_logs.append(f"Matrix '{name}' does not have an inverse.")
            return (no_update,) * 6 + (new_output_logs,)

        inverse_name = generate_inverse_matrix_name(
            name=name, existing_names=list(stored_matrices.keys())
        )

        stored_matrices[inverse_name] = inverted_matrix.tolist()
        previous_vectors.append(stored_vectors.copy())
        new_vectors = apply_matrix_to_vectors(inverted_matrix, stored_vectors)

        most_recent_matrix = np.array(stored_matrices[inverse_name])
        new_steps = app_instance.update_animations(
            animation_steps=animation_steps, end_matrix=most_recent_matrix
        )

        return (
            stored_matrices,
            # create_figure(new_vectors),
            new_vectors,
            previous_vectors,
            {},  # Empty undone matrices.
            False,
            new_steps,
            new_output_logs,
        )

    @app_instance.app.callback(
        Output("matrix-store", "data", allow_duplicate=True),
        # Output("graph", "figure", allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        Output("undone-matrices-store", "data", allow_duplicate=True),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("animation-undo-mode", "data", allow_duplicate=True),
        Output("animation-steps", "data", allow_duplicate=True),
        Output("output-logs", "data", allow_duplicate=True),
        [
            Input({"type": "interactable", "name": "undo-matrix-button"}, "n_clicks"),
            State("matrix-store", "data"),
        ],
        [
            State("vector-store", "data"),
            State("previous-vector-store", "data"),
            State("undone-matrices-store", "data"),
            State("output-logs", "data"),
        ],
        [
            State("animation-steps", "data"),
        ],
        prevent_initial_call=True,
    )
    def undo_matrix(
        _,
        stored_matrices: MatrixDict,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        undone_matrices: MatrixDict,
        output_logs: list[str],
        animation_steps: list[Matrix],
    ) -> tuple:
        if not stored_matrices:
            output_logs.append("Undo Matrix: No matrices exist.")
            return (no_update,) * 7 + (output_logs,)

        new_stored_matrices = stored_matrices.copy()
        new_stored_vectors = stored_vectors.copy()
        new_previous_vectors = previous_vectors.copy()
        new_undone_matrices = undone_matrices.copy()

        last_matrix_name = list(new_stored_matrices.keys())[-1]
        last_matrix = np.array(new_stored_matrices[last_matrix_name])
        inverse_matrix = safe_inverse(last_matrix)

        # This is done so that it doesn't delete any vectors that were made
        # before the undoing.
        new_previous_vectors, output_log_updates = _handle_newly_added_vectors(
            new_stored_vectors, new_previous_vectors, inverse_matrix
        )

        new_undone_matrices[last_matrix_name] = new_stored_matrices.pop(
            last_matrix_name
        )

        restored_vectors = new_previous_vectors.pop()

        # This lets us play the last animation backwards without manually
        # reversing the animation_steps. Playing said animation backwards
        # also ignores any quirks that might happen with uninvertible
        # matrices.
        new_steps = app_instance.update_animations(
            animation_steps=animation_steps.copy(),
            start_matrix=last_matrix,
            end_matrix=app_instance.identity,
        )
        animations_disabled = False
        undo_mode = True

        new_output_logs = no_update
        if output_log_updates is not None:
            new_output_logs = output_logs + [output_log_updates]

        return (
            new_stored_matrices,
            restored_vectors,
            new_previous_vectors,
            new_undone_matrices,
            animations_disabled,
            undo_mode,
            new_steps,
            new_output_logs,
        )

    @app_instance.app.callback(
        Output("matrix-store", "data", allow_duplicate=True),
        # Output('graph', 'figure', allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        Output("undone-matrices-store", "data", allow_duplicate=True),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("animation-steps", "data", allow_duplicate=True),
        Output("output-logs", "data", allow_duplicate=True),
        [
            Input({"type": "interactable", "name": "redo-matrix-button"}, "n_clicks"),
            State("matrix-store", "data"),
        ],
        [
            State("vector-store", "data"),
            State("previous-vector-store", "data"),
            State("undone-matrices-store", "data"),
            State("output-logs", "data"),
        ],
        [State("animation-steps", "data")],
        prevent_initial_call=True,
    )
    def redo_matrix(
        _,
        stored_matrices: MatrixDict,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        undone_matrices: MatrixDict,
        output_logs: list[str],
        animation_steps: list[Matrix],
    ) -> tuple:
        # A condition to check for an empty `stored_matrices` is
        # not needed because `undone_matrices` may not be empty
        # while `stored_matrices` is empty, but if `undone_matrices`
        # is empty, then `stored_matrices` is for sure empty too.
        if not undone_matrices:
            output_logs.append("No matrices to redo.")
            return (no_update,) * 6 + (output_logs,)

        last_undone_matrix_name = list(undone_matrices.keys())[-1]
        stored_matrices[last_undone_matrix_name] = undone_matrices.pop(
            last_undone_matrix_name
        )

        previous_vectors.append(stored_vectors.copy())

        most_recent_matrix = np.array(stored_matrices[last_undone_matrix_name])
        restored_vectors = apply_matrix_to_vectors(most_recent_matrix, stored_vectors)

        new_steps = app_instance.update_animations(
            animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix
        )

        return (
            stored_matrices,
            # create_figure(restored_vectors),
            restored_vectors,
            previous_vectors,
            undone_matrices,
            False,
            new_steps,
            no_update,
        )

    @app_instance.app.callback(
        Output("matrix-store", "data"),
        # Output('graph', 'figure'),
        Output("vector-store", "data"),
        Output("previous-vector-store", "data"),
        Output("undone-matrices-store", "data"),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("animation-steps", "data", allow_duplicate=True),
        Output("output-logs", "data"),
        [
            Input({"type": "interactable", "name": "repeat-matrix-button"}, "n_clicks"),
            State(
                {"type": "interactable", "name": "repeat-matrix-entry-name"},
                "value",
            ),
        ],
        [
            State("matrix-store", "data"),
            State("vector-store", "data"),
            State("previous-vector-store", "data"),
            State("output-logs", "data"),
        ],
        [State("animation-steps", "data")],
        prevent_initial_call=True,
    )
    def repeat_matrix(
        _,
        selected_matrix: str | None,
        stored_matrices: MatrixDict,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        output_logs: list[str],
        animation_steps: list[Matrix],
    ) -> tuple:
        def _validate_input(
            name: str | None,
            stored_matrices_: MatrixDict | tuple | list,
            output_logs_: list[str],
        ) -> tuple[bool, list[str]]:
            if not stored_matrices_:
                output_logs_.append("Repeat Matrix: No matrices exist.")
                return False, output_logs_
            if name and (name not in stored_matrices_):
                output_logs_.append(f"Repeat Matrix: Matrix '{name}' does not exist.")
                return False, output_logs_
            return True, output_logs_

        valid, new_output_logs = _validate_input(
            name=selected_matrix,
            stored_matrices_=stored_matrices,
            output_logs_=output_logs,
        )
        if not valid:
            return (no_update,) * 6 + (new_output_logs,)

        if not selected_matrix:
            selected_matrix = list(stored_matrices.keys())[-1]

        new_name = generate_duplicate_matrix_name(
            selected_matrix, list(stored_matrices.keys())
        )

        stored_matrices[new_name] = stored_matrices[selected_matrix]
        previous_vectors.append(stored_vectors.copy())
        new_vectors = apply_matrix_to_vectors(
            stored_matrices[selected_matrix], stored_vectors
        )

        most_recent_matrix = np.array(list(stored_matrices.values())[-1])
        new_steps = app_instance.update_animations(
            animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix
        )

        return (
            stored_matrices,
            # create_figure(new_vectors),
            new_vectors,
            previous_vectors,
            {},  # Empty undone matrices.
            False,
            new_steps,
            no_update,
        )
