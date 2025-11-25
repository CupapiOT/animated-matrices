from dash.dependencies import Input, Output, State
from src.config.constants import LOWER_LETTERS
from src.utils.general import set_nonetype_to_zero
from src.utils.matrix import safe_inverse
from src.utils.vector import calculate_longest_vector_mag
from src.types import MatrixDict, Vectors, Number
from src.graph_functions.create_figures import create_figure
import numpy as np


def _update_all_previous_vectors(
    stored_matrices: MatrixDict,
    vector_name: str,
    x: Number,
    y: Number,
    color: str,
    previous_vectors: list[Vectors],
    output_logs: list[str],
) -> tuple[list[Vectors], list[str]]:
    """Only used within the `add_or_edit_vector` method as a
    place to refactor code from `add_or_edit_vector`, which is
    defined in `self.register_callback`, directly below this
    function.
    """

    new_output_logs = output_logs
    most_to_least_recent_matrices = dict(reversed(stored_matrices.items()))
    matrices = {
        name: np.array(mat) for name, mat in most_to_least_recent_matrices.items()
    }
    most_to_least_recent_prev_vecs = list(reversed(previous_vectors.copy()))
    new_previous_vectors = most_to_least_recent_prev_vecs.copy()
    previous_vectors_temp = None
    for vectors, (its_matrixs_name, its_matrix) in zip(
        new_previous_vectors, matrices.items()
    ):
        if vector_name not in vectors:
            break
        inverse_matrix = safe_inverse(its_matrix)
        if inverse_matrix is not None:
            edited_vector = (
                np.array([x, y])
                if (previous_vectors_temp is None)
                else (previous_vectors_temp)
            )
            inverted_edited_vector_vals = inverse_matrix @ edited_vector
            previous_vectors_temp = inverted_edited_vector_vals.copy()
            inverted_edited_vector = [
                inverted_edited_vector_vals.tolist(),
                color,
            ]
            vectors[vector_name] = inverted_edited_vector
        else:
            if previous_vectors_temp is not None:
                previous_vectors_temp_vector = [
                    previous_vectors_temp.tolist(),
                    color,
                ]
            else:
                previous_vectors_temp_vector = None
            edited_vector = (
                [(x, y), color]
                if (previous_vectors_temp_vector is None)
                else (previous_vectors_temp_vector)
            )
            vectors[vector_name] = edited_vector

            new_output_logs.append(
                f'Edited vector "{vector_name}" was unable to be '
                f"properly shown before the matrix "
                f'"{its_matrixs_name}" was applied due to it being '
                f"singular. "
            )

    return new_previous_vectors, new_output_logs


def register_vector_callbacks(app_instance):
    @app_instance.app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        Output("output-logs", "data", allow_duplicate=True),
        [
            Input({"type": "interactable", "name": "add-vector-button"}, "n_clicks"),
            State({"type": "interactable", "name": "vector-entry-1"}, "value"),
            State({"type": "interactable", "name": "vector-entry-2"}, "value"),
            State({"type": "interactable", "name": "vector-entry-color"}, "value"),
            State("vector-store", "data"),
            State({"type": "interactable", "name": "new-vector-entry-name"}, "value"),
        ],
        # Separated list to indicate that the below is for a
        # different part of the function.
        [
            State("matrix-store", "data"),
            State("previous-vector-store", "data"),
            State("output-logs", "data"),
        ],
        prevent_initial_call=True,
    )
    def add_or_edit_vector(
        n_clicks: int,
        x_val: Number,
        y_val: Number,
        color: str,
        stored_vectors: Vectors,
        name: str,
        stored_matrices: MatrixDict,
        previous_vectors: list[Vectors],
        output_logs: list[str],
    ) -> tuple:
        try:
            x, y = map(float, set_nonetype_to_zero(x_val, y_val))
        except (ValueError, TypeError):
            x, y = 0, 0

        vector_name = name if name else (LOWER_LETTERS[n_clicks % 26 - 1])
        stored_vectors[vector_name] = [(x, y), color]
        if not previous_vectors:
            return (
                create_figure(
                    vectors=stored_vectors,
                    scale=(calculate_longest_vector_mag(stored_vectors) * 1.1),
                ),
                stored_vectors,
                [],  # Empty list of undone-vectors.
                output_logs,
            )

        # This is done so that any recently edited vectors are kept
        # visually consistent after any matrix-undo-s.
        new_previous_vectors, new_output_logs = _update_all_previous_vectors(
            stored_matrices, vector_name, x, y, color, previous_vectors, output_logs
        )

        return (
            create_figure(
                vectors=stored_vectors,
                scale=(calculate_longest_vector_mag(stored_vectors) * 1.1),
            ),
            stored_vectors,
            list(reversed(new_previous_vectors)),
            new_output_logs,
        )

    @app_instance.app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("vector-store", "data", allow_duplicate=True),
        Output("previous-vector-store", "data", allow_duplicate=True),
        [
            Input({"type": "interactable", "name": "delete-vector-button"}, "n_clicks"),
            State(
                {"type": "interactable", "name": "delete-vector-entry-name"},
                "value",
            ),
        ],
        # This time the separated list to indicate that the below
        # is for not for the main inputs of the function.
        [State("vector-store", "data"), State("previous-vector-store", "data")],
        prevent_initial_call=True,
    )
    def delete_vector(
        _,
        name: str,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
    ) -> tuple:
        if not name:
            name = list(stored_vectors.keys())[-1]
        if name not in stored_vectors:
            return (
                create_figure(
                    vectors=stored_vectors,
                    scale=calculate_longest_vector_mag(stored_vectors) * 1.1,
                ),
                stored_vectors,
                previous_vectors,
            )

        del stored_vectors[name]
        for vectors in previous_vectors:
            try:
                del vectors[name]
            except KeyError:  # Doesn't matter, just keep deleting.
                continue

        return (
            create_figure(
                vectors=stored_vectors,
                scale=calculate_longest_vector_mag(stored_vectors) * 1.1,
            ),
            stored_vectors,
            previous_vectors,
        )
