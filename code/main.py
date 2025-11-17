from dash.exceptions import PreventUpdate
import numpy as np
from dash import Dash, callback_context, _callback, no_update, ALL, html
import dash_bootstrap_components as dbc
import dash_latex as dl
from dash.dependencies import Input, Output, State
import re
from constants import *
from create_figures import create_figure
from project_types import *
from matrix_utils import safe_inverse
from general_utils import set_nonetype_to_zero
from layout import create_layout


class MatrixTransformationsApp:
    def __init__(self, basis_vectors):
        self.app = Dash(
            title="Matrix Transformations",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=META_TAGS,
        )

        self.BASIS_VECTORS = basis_vectors

        # Potentially able to change this later, for different coordinate systems.
        self.identity: np.ndarray = np.identity(2)

        # Animation timing; implementable via dcc.store in the future.
        FRAMES_PER_SECOND: int = 12
        TIME_FOR_ANIMATION_MS: int = 1000
        self.animation_frames_count: int = FRAMES_PER_SECOND * (
            TIME_FOR_ANIMATION_MS // 1000
        )
        interval_ms = TIME_FOR_ANIMATION_MS / self.animation_frames_count
        self.interval_ms = max(int(interval_ms), 1)  # Always at least 1ms

        self.app.layout = create_layout(self)
        self._register_callbacks()

    def _handle_newly_added_vectors(
        self,
        stored_vectors: Vectors,
        previous_vectors: list[Vectors],
        inverse_matrix: Matrix | None,
    ) -> tuple[list[Vectors], str | None]:
        """Only used within the undo_matrix method, which is defined
        in `self.register_callback`."""
        if len(stored_vectors) <= len(previous_vectors[-1]):
            return previous_vectors, None

        new_output_log = None
        new_previous_vectors = previous_vectors.copy()
        new_keys = set(stored_vectors) - (set(new_previous_vectors[-1]))
        new_vector_dict = {key: stored_vectors[key] for key in new_keys}
        if inverse_matrix is not None:
            inverted_new_vectors = self.apply_matrix_to_vectors(
                inverse_matrix, new_vector_dict
            )
            new_previous_vectors[-1].update(inverted_new_vectors)
        else:
            new_previous_vectors[-1].update(new_vector_dict)
            new_output_log = (
                f"The coordinates of newly added vector(s) {list(new_keys)} "
                f"were not updated due to how the previous matrix was unable "
                f"to be inverted. "
            )

        return new_previous_vectors, new_output_log

    def _register_callbacks(self) -> None:
        def _vector_getter(x_val: Number, y_val: Number) -> tuple[Number, Number]:
            try:
                x, y = map(float, set_nonetype_to_zero(x_val, y_val))
            except (ValueError, TypeError):
                x, y = 0, 0
            return x, y

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
                name: np.array(mat)
                for name, mat in most_to_least_recent_matrices.items()
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

        @self.app.callback(
            Output("graph", "figure", allow_duplicate=True),
            Output("vector-store", "data", allow_duplicate=True),
            Output("previous-vector-store", "data", allow_duplicate=True),
            Output("output-logs", "data", allow_duplicate=True),
            [
                Input(
                    {"type": "interactable", "name": "add-vector-button"}, "n_clicks"
                ),
                State({"type": "interactable", "name": "vector-entry-1"}, "value"),
                State({"type": "interactable", "name": "vector-entry-2"}, "value"),
                State({"type": "interactable", "name": "vector-entry-color"}, "value"),
                State("vector-store", "data"),
                State(
                    {"type": "interactable", "name": "new-vector-entry-name"}, "value"
                ),
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
            x, y = _vector_getter(x_val, y_val)
            vector_name = name if name else (LOWER_LETTERS[n_clicks % 26 - 1])
            stored_vectors[vector_name] = [(x, y), color]
            if not previous_vectors:
                return (
                    create_figure(
                        vectors=stored_vectors,
                        scale=(self.calculate_longest_vector_mag(stored_vectors) * 1.1),
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
                    scale=(self.calculate_longest_vector_mag(stored_vectors) * 1.1),
                ),
                stored_vectors,
                list(reversed(new_previous_vectors)),
                new_output_logs,
            )

        @self.app.callback(
            Output("graph", "figure", allow_duplicate=True),
            Output("vector-store", "data", allow_duplicate=True),
            Output("previous-vector-store", "data", allow_duplicate=True),
            [
                Input(
                    {"type": "interactable", "name": "delete-vector-button"}, "n_clicks"
                ),
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
                        scale=self.calculate_longest_vector_mag(stored_vectors) * 1.1,
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
                    scale=self.calculate_longest_vector_mag(stored_vectors) * 1.1,
                ),
                stored_vectors,
                previous_vectors,
            )

        def create_frames(
            end_matrix: np.ndarray,
            start_matrix: np.ndarray = self.identity,
            steps: int = self.animation_frames_count,
        ) -> list[Matrix]:
            """
            Creates interpolation frames from one matrix to another.
            Parameters:
            - start_matrix: Any matrix. Usually the identity matrix, details below.
            - end_matrix: Any matrix.
            - steps: The number of interpolated frames to return.
            Returns:
            - A list of matrices containing the interpolated matrices.

            IMPORTANT:
              When generating animation frames, we interpolate from the identity matrix
              to the matrix being applied. This is because the animation applies each
              intermediate matrix to the CURRENT positions of the vectors on the graph.

              This means we must not interpolate from the last matrix applied.
              Doing so creates either:
              - a zero-difference (no animation if matrices are equal), or
              - unintended compound transformations (exponential growth when chaining).

              Always interpolate from the intended identity matrix to the intended matrix.
            """

            # The first frame is also returned for future compatibility for
            # exporting animations.
            return [
                (1 - t) * start_matrix + t * end_matrix
                for t in np.linspace(0, 1, num=steps + 1)
            ]

        def update_animations(
            animation_steps: list[Matrix],
            end_matrix: np.ndarray,
            start_matrix: np.ndarray = self.identity,
            steps: int = self.animation_frames_count,
        ) -> list[Matrix]:
            """Returns animation_steps + new_frames."""
            frames = create_frames(
                end_matrix=end_matrix, start_matrix=start_matrix, steps=steps
            )
            new_steps = animation_steps + frames
            return new_steps

        @self.app.callback(
            Output("matrix-store", "data", allow_duplicate=True),
            # Output('graph', 'figure', allow_duplicate=True),
            Output("vector-store", "data", allow_duplicate=True),
            Output("previous-vector-store", "data", allow_duplicate=True),
            Output("undone-matrices-store", "data", allow_duplicate=True),
            Output("animation-interval", "disabled", allow_duplicate=True),
            Output("animation-steps", "data", allow_duplicate=True),
            [
                Input(
                    {"type": "interactable", "name": "add-matrix-button"}, "n_clicks"
                ),
                State({"type": "interactable", "name": "matrix-entry-1"}, "value"),
                State({"type": "interactable", "name": "matrix-entry-2"}, "value"),
                State({"type": "interactable", "name": "matrix-entry-3"}, "value"),
                State({"type": "interactable", "name": "matrix-entry-4"}, "value"),
            ],
            [
                State("matrix-store", "data"),
                State("vector-store", "data"),
                State("previous-vector-store", "data"),
                State(
                    {"type": "interactable", "name": "new-matrix-entry-name"}, "value"
                ),
            ],
            [State("animation-steps", "data")],
            prevent_initial_call=True,
        )
        def add_matrix(
            n_clicks: int,
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
            matrix_name = name if name else UPPER_LETTERS[n_clicks % 26 - 1]

            stored_matrices[matrix_name] = matrix.tolist()

            previous_vectors.append(stored_vectors.copy())

            most_recent_matrix = np.array(list(stored_matrices.values())[-1])
            new_vectors = self.apply_matrix_to_vectors(
                most_recent_matrix, stored_vectors
            )

            new_steps = update_animations(
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

        @self.app.callback(
            Output("graph", "figure", allow_duplicate=True),
            Output("animation-interval", "disabled", allow_duplicate=True),
            Output("animation-undo-mode", "data", allow_duplicate=True),
            Output("animation-steps", "data", allow_duplicate=True),
            [
                Input("animation-interval", "n_intervals"),
                State("animation-undo-mode", "data"),
                State("animation-steps", "data"),
            ],
            [
                State("previous-vector-store", "data"),
                State("vector-store", "data"),
                State("undone-matrices-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def animate_graph(
            _,  # Ignore `n_intervals` of animation-interval
            undo_mode: bool,
            animation_steps: list[Matrix],
            previous_vectors: list[Vectors],
            stored_vectors: Vectors,
            undone_matrices: MatrixDict,
        ) -> tuple:
            if not animation_steps:
                return (
                    no_update,
                    True,
                    False,
                    no_update,
                )

            if undo_mode:
                vectors_to_animate = stored_vectors
                last_undone_matrix_name = list(undone_matrices.keys())[-1]
                last_undone_matrix = undone_matrices[last_undone_matrix_name]
                last_frame_vectors = self.apply_matrix_to_vectors(
                    last_undone_matrix, stored_vectors
                )
            else:
                try:
                    vectors_to_animate = previous_vectors[-1]
                except IndexError:
                    vectors_to_animate = self.BASIS_VECTORS
                last_frame_vectors = stored_vectors

            current_frame = animation_steps[0]
            interpolated_vectors = self.apply_matrix_to_vectors(
                current_frame, vectors_to_animate
            )

            # Get the appropriate scale of the graph.
            first_frame_mag = self.calculate_longest_vector_mag(vectors_to_animate)
            last_frame_mag = self.calculate_longest_vector_mag(last_frame_vectors)
            graph_scale = max(first_frame_mag, last_frame_mag) * 1.1

            return (
                create_figure(vectors=interpolated_vectors, scale=graph_scale),
                no_update,
                no_update,
                animation_steps[1:] if animation_steps else [],
            )

        @self.app.callback(
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

        def generate_unique_matrix_name(name: str, existing_names) -> str:
            def _remove_duplicate_suffix(base_name: str) -> str:
                non_duplicate_name = base_name
                # A name is a duplicate if it ends in ` (<int>)`.
                name_is_duplicate = re.search(r" \((\d+)\)$", base_name) is not None
                if name_is_duplicate:
                    non_duplicate_name = base_name[:-4]
                return non_duplicate_name

            def _find_next_num_in_sequence(names):
                existing_duplicates = [
                    name for name in names if name.startswith(new_name)
                ]
                number_list = sorted(
                    [
                        int(re.search(r" \((\d+)\)$", duplicate_name).group(1))  # type: ignore
                        for duplicate_name in existing_duplicates
                        if re.search(r" \((\d+)\)$", duplicate_name)
                    ]
                )
                try:
                    next_num = next(
                        (
                            number_list[i] + 1
                            for i in range(len(number_list) - 1)
                            if number_list[i + 1] != number_list[i] + 1
                        ),
                        number_list[-1] + 1,
                    )
                except IndexError:
                    next_num = 2
                return next_num

            if name not in existing_names:
                return name
            new_name = _remove_duplicate_suffix(name)
            index_solution = _find_next_num_in_sequence(existing_names)
            return new_name + f" ({index_solution})"

        @self.app.callback(
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
                    key for key in stored_matrices_.keys() if not key.startswith("I_")
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

            inverse_name = "I_{" + name + "}"
            # TODO: Create better name handling appropriate for math expressions.
            # if "^{-1}" in name:
            #     inverse_name = "(" + name + ")^{-1}"
            # else:
            #     inverse_name = name + r"^{-1}"
            new_name = generate_unique_matrix_name(
                name=inverse_name, existing_names=stored_matrices
            )

            stored_matrices[new_name] = inverted_matrix.tolist()
            previous_vectors.append(stored_vectors.copy())
            new_vectors = self.apply_matrix_to_vectors(inverted_matrix, stored_vectors)

            most_recent_matrix = np.array(stored_matrices[new_name])
            new_steps = update_animations(
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

        @self.app.callback(
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
                Input(
                    {"type": "interactable", "name": "undo-matrix-button"}, "n_clicks"
                ),
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
            new_previous_vectors, output_log_updates = self._handle_newly_added_vectors(
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
            new_steps = update_animations(
                animation_steps=animation_steps.copy(),
                start_matrix=last_matrix,
                end_matrix=self.identity,
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

        @self.app.callback(
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
                    {"type": "interactable", "name": "redo-matrix-button"}, "n_clicks"
                ),
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
            restored_vectors = self.apply_matrix_to_vectors(
                most_recent_matrix, stored_vectors
            )

            new_steps = update_animations(
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

        @self.app.callback(
            Output("matrix-store", "data"),
            # Output('graph', 'figure'),
            Output("vector-store", "data"),
            Output("previous-vector-store", "data"),
            Output("undone-matrices-store", "data"),
            Output("animation-interval", "disabled", allow_duplicate=True),
            Output("animation-steps", "data", allow_duplicate=True),
            Output("output-logs", "data"),
            [
                Input(
                    {"type": "interactable", "name": "repeat-matrix-button"}, "n_clicks"
                ),
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
                print(output_logs_)
                if not stored_matrices_:
                    output_logs_.append("Repeat Matrix: No matrices exist.")
                    return False, output_logs_
                if name and (name not in stored_matrices_):
                    output_logs_.append(
                        f"Repeat Matrix: Matrix '{name}' does not exist."
                    )
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

            new_name = generate_unique_matrix_name(selected_matrix, stored_matrices)

            stored_matrices[new_name] = stored_matrices[selected_matrix]
            previous_vectors.append(stored_vectors.copy())
            new_vectors = self.apply_matrix_to_vectors(
                stored_matrices[selected_matrix], stored_vectors
            )

            most_recent_matrix = np.array(list(stored_matrices.values())[-1])
            new_steps = update_animations(
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

        @self.app.callback(
            Output({"type": "interactable", "name": "add-vector-button"}, "children"),
            [
                Input(
                    {"type": "interactable", "name": "new-vector-entry-name"}, "value"
                ),
                Input("vector-store", "data"),
                State(
                    {"type": "interactable", "name": "add-vector-button"}, "n_clicks"
                ),
            ],
            prevent_initial_call=True,
        )
        def _change_vector_button_name(
            name_input: str, vectors: Vectors, n_clicks: int
        ) -> str:
            names = [name for name in vectors]
            if name_input in names or LOWER_LETTERS[n_clicks % 26] in names:
                return "Edit Vector"
            else:
                return "Add Vector"

        @self.app.callback(
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

        @self.app.callback(
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

    @staticmethod
    def calculate_longest_vector_mag(vectors: Vectors) -> Number:
        magnitude_of_longest_vector: Number = -float("inf")
        for (vector_coords), _ in vectors.values():
            vector_mag = np.linalg.norm(np.array(vector_coords))
            magnitude_of_longest_vector = max(
                magnitude_of_longest_vector, float(vector_mag)
            )
        return magnitude_of_longest_vector

    @staticmethod
    def apply_matrix_to_vectors(matrix: Matrix, vectors: Vectors) -> Vectors:
        new_vectors = vectors.copy()
        vector_list = [np.array([x, y]) for ((x, y), _) in new_vectors.values()]
        transformed_vectors = [(matrix @ vector).tolist() for vector in vector_list]

        for (name, (_, color)), t_vector in zip(
            new_vectors.items(), transformed_vectors
        ):
            new_vectors[name] = [t_vector, color]

        return new_vectors


def main() -> None:
    app = MatrixTransformationsApp(
        {"i-hat": [(1, 0), "green"], "j-hat": [(0, 1), "red"]}
    )
    app.app.run(debug=True)


if __name__ == "__main__":
    main()
