import numpy as np
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State
import re
from constants import *
from create_figures import create_2d_basis_vectors, create_figure
from project_types import *
from matrix_utils import safe_inverse
from general_utils import set_nonetype_to_zero


class MatrixTransformationsApp:
    def __init__(self, basis_vectors):
        self.app = Dash('Matrix Transformations')

        self.BASIS_VECTORS = basis_vectors

        # Potentially able to change this later, for different coordinate systems.
        self.identity: np.ndarray = np.identity(2)

        # Animation timing; implementable via dcc.store in the future.
        FRAMES_PER_SECOND: int = 12
        TIME_FOR_ANIMATION_MS: int = 1000
        self.animation_frames_count: int = FRAMES_PER_SECOND * (TIME_FOR_ANIMATION_MS // 1000)
        interval_ms = TIME_FOR_ANIMATION_MS / self.animation_frames_count
        self.interval_ms = max(int(interval_ms), 1)  # Always at least 1ms

        self.app.layout = self._create_layout()
        self._register_callbacks()

    def _handle_newly_added_vectors(
            self,
            stored_vectors: Vectors,
            previous_vectors: list[Vectors],
            inverse_matrix: Matrix | None
    ) -> tuple[list[Vectors], str]:
        """Only used within the undo_matrix method, which is defined
        in `self.register_callback`."""
        if len(stored_vectors) <= len(previous_vectors[-1]):
            return previous_vectors, ''

        new_output_log = ''
        new_previous_vectors = previous_vectors.copy()
        new_keys = set(stored_vectors) - (set(new_previous_vectors[-1]))
        new_vector_dict = {key: stored_vectors[key]
                           for key in new_keys}
        if inverse_matrix is not None:
            inverted_new_vectors = self.apply_matrix_to_vectors(
                inverse_matrix,
                new_vector_dict
            )
            new_previous_vectors[-1].update(inverted_new_vectors)
        else:
            new_previous_vectors[-1].update(new_vector_dict)
            new_output_log += (
                f'Newly added vector(s) {list(new_keys)} were not '
                f'changed due to how the previous matrix was unable '
                f'to be inverted. '
            )

        return new_previous_vectors, new_output_log

    def _register_callbacks(self) -> None:
        def _vector_getter(
                x_val: Number,
                y_val: Number
        ) -> tuple[Number, Number]:
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
                output_logs: str
        ) -> tuple[list[Vectors], str]:
            """Only used within the `add_or_edit_vector` method as a
            place to refactor code from `add_or_edit_vector`, which is
            defined in `self.register_callback`, directly below this
            function.
            """

            new_output_logs = output_logs
            most_to_least_recent_matrices = dict(
                reversed(stored_matrices.items())
            )
            matrices = {name: np.array(mat)
                        for name, mat in most_to_least_recent_matrices.items()}
            most_to_least_recent_prev_vecs = list(
                reversed(previous_vectors.copy())
            )
            new_previous_vectors = most_to_least_recent_prev_vecs.copy()
            previous_vectors_temp = None
            for vectors, (its_matrixs_name, its_matrix) in zip(
                    new_previous_vectors,
                    matrices.items()
            ):
                if vector_name not in vectors:
                    break
                inverse_matrix = safe_inverse(its_matrix)
                if inverse_matrix is not None:
                    edited_vector = np.array([x, y]) if (
                            previous_vectors_temp is None) else (
                        previous_vectors_temp)
                    inverted_edited_vector_vals = (
                            inverse_matrix @ edited_vector)
                    previous_vectors_temp = inverted_edited_vector_vals.copy()
                    inverted_edited_vector = [
                        inverted_edited_vector_vals.tolist(),
                        color
                    ]
                    vectors[vector_name] = inverted_edited_vector
                else:
                    if previous_vectors_temp is not None:
                        previous_vectors_temp_vector = [
                            previous_vectors_temp.tolist(),
                            color
                        ]
                    else:
                        previous_vectors_temp_vector = None
                    edited_vector = [(x, y), color] if (
                            previous_vectors_temp_vector is None) else (
                        previous_vectors_temp_vector)
                    vectors[vector_name] = edited_vector

                    new_output_logs += (
                        f'Edited vector "{vector_name}" was unable to be '
                        f'properly shown before the matrix '
                        f'"{its_matrixs_name}" was applied due to it being '
                        f'singular. ')

            return new_previous_vectors, new_output_logs

        @self.app.callback(
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            Output('output-logs', 'children', allow_duplicate=True),
            [Input('add-vector-button', 'n_clicks'),
             State('vector-entry-1', 'value'),
             State('vector-entry-2', 'value'),
             State('vector-entry-color', 'value'),
             State('vector-store', 'data'),
             State('new-vector-entry-name', 'value'),
             ],
            # Separated list to indicate that the below is for a
            # different part of the function.
            [State('matrix-store', 'data'),
             State('previous-vector-store', 'data'),
             State('output-logs', 'children')
             ],
            prevent_initial_call=True
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
                output_logs: str
        ) -> tuple:
            x, y = _vector_getter(x_val, y_val)
            vector_name = name if name else (LOWER_LETTERS[n_clicks % 26 - 1])
            stored_vectors[vector_name] = [(x, y), color]
            if not previous_vectors:
                return (create_figure(stored_vectors),
                        stored_vectors,
                        [],
                        output_logs)

            # This is done so that any recently edited vectors are kept
            # visually consistent after the undo.
            new_previous_vectors, new_output_logs = (
                _update_all_previous_vectors(
                    stored_matrices,
                    vector_name,
                    x,
                    y,
                    color,
                    previous_vectors,
                    output_logs
                ))

            return (create_figure(stored_vectors),
                    stored_vectors,
                    list(reversed(new_previous_vectors)),
                    new_output_logs)

        @self.app.callback(
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            [Input('delete-vector-button', 'n_clicks'),
             State('delete-vector-entry-name', 'value')
             ],
            # This time the separated list to indicate that the below
            # is for not for the main inputs of the function.
            [State('vector-store', 'data'),
             State('previous-vector-store', 'data')
             ],
            prevent_initial_call=True
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
                return (create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors)

            del stored_vectors[name]
            for vectors in previous_vectors:
                try:
                    del vectors[name]
                except KeyError:  # Doesn't matter, just keep deleting.
                    continue

            return (create_figure(stored_vectors),
                    stored_vectors,
                    previous_vectors)

        def create_frames(end_matrix: np.ndarray, start_matrix: np.ndarray = self.identity, steps: int = self.animation_frames_count) -> list[Matrix]:
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

            # The first frame is also returned due to it playing an important role
            # in determining what to animate on the graph.
            return [(1 - t) * start_matrix + t * end_matrix
                    for t in np.linspace(0, 1, num=steps + 1)]

        def update_animations(animation_steps: list[Matrix], end_matrix: np.ndarray, start_matrix: np.ndarray = self.identity, steps: int = self.animation_frames_count) -> list[Matrix]:
            """Returns animation_steps + new_frames."""
            frames = create_frames(end_matrix=end_matrix, start_matrix=start_matrix, steps=steps)
            new_steps = animation_steps + frames
            return new_steps

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            # Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            Output('animation-interval', 'disabled', allow_duplicate=True),
            Output('animation-steps', 'data', allow_duplicate=True),
            [Input('add-matrix-button', 'n_clicks'),
             State('matrix-entry-1', 'value'),
             State('matrix-entry-2', 'value'),
             State('matrix-entry-3', 'value'),
             State('matrix-entry-4', 'value')
             ],
            [State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vector-store', 'data'),
             State('new-matrix-entry-name', 'value'),
             ],
            [State('animation-steps', 'data')
             ],
            prevent_initial_call=True
        )
        def add_matrix(
                n_clicks: int,
                a: Number, b: Number, c: Number, d: Number,
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
                most_recent_matrix,
                stored_vectors
            )

            new_steps = update_animations(animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix)

            return (stored_matrices,
                    str(stored_matrices),
                    # create_figure(restored_vectors),
                    new_vectors,
                    previous_vectors,
                    {},
                    False,
                    new_steps)

        @self.app.callback(
            Output('graph', 'figure', allow_duplicate=True),
            Output('animation-interval', 'disabled'),
            Output('animation-steps', 'data'),
            [Input('animation-interval', 'n_intervals'),
             State('animation-steps', 'data')
             ],
            [State('previous-vector-store', 'data')
             ],
            prevent_initial_call=True
        )
        def animate_graph(
                can_animate: bool,
                animation_steps: list[Matrix],
                previous_vectors: list[Vectors]
        ) -> tuple:
            if (not animation_steps) or (not can_animate):
                return no_update, True, no_update

            # Reduces lagging when adding more animations before the current one
            # is finished.
            vectors_to_animate = previous_vectors[-1]
            identity_count = sum(np.array_equal(matrix, self.identity) for matrix in animation_steps)
            if identity_count >= 1 and not np.array_equal(animation_steps[0], self.identity):
                vectors_to_animate = previous_vectors[-(identity_count + 1)]

            current_frame = animation_steps[0]
            interpolated_vectors = self.apply_matrix_to_vectors(
                current_frame,
                vectors_to_animate
            )
            return (create_figure(interpolated_vectors),
                    no_update,
                    animation_steps[1:] if animation_steps else [])

        def generate_unique_matrix_name(name: str, existing_names) -> str:
            def _remove_duplicate_suffix(base_name: str) -> str:
                non_duplicate_name = base_name
                # A name is a duplicate if it ends in ` (<int>)`.
                name_is_duplicate = (
                        re.search(r' \((\d+)\)$', base_name) is not None
                )
                if name_is_duplicate:
                    non_duplicate_name = base_name[:-4]
                return non_duplicate_name

            def _find_next_num_in_sequence(names):
                existing_duplicates = [
                    name for name in names
                    if name.startswith(new_name)
                ]
                number_list = sorted([
                    int(re.search(r' \((\d+)\)$', duplicate_name).group(1))
                    for duplicate_name in existing_duplicates
                    if re.search(r' \((\d+)\)$', duplicate_name)
                ])
                try:
                    next_num = next(
                        (number_list[i] + 1
                         for i in range(len(number_list) - 1)
                         if number_list[i + 1] != number_list[i] + 1),
                        number_list[-1] + 1
                    )
                except IndexError:
                    next_num = 2
                return next_num

            if name not in existing_names:
                return name
            new_name = _remove_duplicate_suffix(name)
            index_solution = _find_next_num_in_sequence(existing_names)
            return new_name + f' ({index_solution})'

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            # Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            Output('output-logs', 'children', allow_duplicate=True),
            Output('animation-interval', 'disabled', allow_duplicate=True),
            Output('animation-steps', 'data', allow_duplicate=True),
            [Input('inverse-matrix-button', 'n_clicks'),
             State('inverse-matrix-entry-name', 'value'),
             ],
            [State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vector-store', 'data'),
             State('undone-matrices-store', 'data'),
             State('output-logs', 'children')
             ],
            [State('animation-steps', 'data')
             ],
            prevent_initial_call=True
        )
        def apply_inverse(
                _,
                matrix_to_invert: str | None,
                stored_matrices: MatrixDict,
                stored_vectors: Vectors,
                previous_vectors: list[Vectors],
                undone_matrices: MatrixDict,
                output_logs: str,
                animation_steps: list[Matrix]
        ) -> tuple:
            def _validate_input(
                    name: str | None,
                    stored_matrices_: MatrixDict | tuple | list,
                    output_logs_: str
            ) -> tuple[bool, str]:
                if not stored_matrices_:
                    output_logs_ += 'No matrices exist. '
                    return False, output_logs_
                if name and (name not in stored_matrices_):
                    output_logs_ += (f'Matrix "{matrix_to_invert}" does not '
                                     f'exist. ')
                    return False, output_logs_

                return True, output_logs_

            def _get_last_matrix_name(
                    stored_matrices_: MatrixDict
            ) -> str:
                non_inverse_matrices = [
                    key for key in stored_matrices_.keys()
                    if not key.startswith('I_')
                ]
                return non_inverse_matrices[-1] if (
                    non_inverse_matrices) else tuple(stored_matrices_)[-1]

            # TODO: REFACTOR WITH no_update
            everything_as_they_are = (
                stored_matrices,
                str(stored_matrices),
                # create_figure(stored_vectors),
                stored_vectors,
                previous_vectors,
                undone_matrices,
            )

            valid, new_output_logs = _validate_input(
                name=matrix_to_invert,
                stored_matrices_=stored_matrices,
                output_logs_=output_logs
            )
            if not valid:
                return everything_as_they_are + (new_output_logs, no_update, no_update)

            name = matrix_to_invert if matrix_to_invert else (
                _get_last_matrix_name(stored_matrices))

            selected_matrix = np.array(stored_matrices[name])
            inverted_matrix = safe_inverse(selected_matrix)
            if inverted_matrix is None:
                log = f'Matrix "{name}" does not have an inverse. '
                new_output_logs += log
                return everything_as_they_are + (new_output_logs, no_update, no_update)

            inverse_name = 'I_' + name
            new_name = generate_unique_matrix_name(
                name=inverse_name,
                existing_names=stored_matrices
            )

            stored_matrices[new_name] = inverted_matrix.tolist()
            previous_vectors.append(stored_vectors.copy())
            new_vectors = self.apply_matrix_to_vectors(
                inverted_matrix,
                stored_vectors
            )

            most_recent_matrix = np.array(stored_matrices[new_name])
            new_steps = update_animations(animation_steps=animation_steps, end_matrix=most_recent_matrix)

            return (stored_matrices,
                    str(stored_matrices),
                    # create_figure(new_vectors),
                    new_vectors,
                    previous_vectors,
                    {},
                    new_output_logs,
                    False,
                    new_steps)

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            Output('output-logs', 'children', allow_duplicate=True),
            [Input('undo-matrix-button', 'n_clicks'),
             State('matrix-store', 'data'),
             ],
            [State('vector-store', 'data'),
             State('previous-vector-store', 'data'),
             State('undone-matrices-store', 'data'),
             State('output-logs', 'children'),
             ],
            prevent_initial_call=True
        )
        def undo_matrix(
                _,
                stored_matrices: MatrixDict,
                stored_vectors: Vectors,
                previous_vectors: list[Vectors],
                undone_matrices: MatrixDict,
                output_logs: str
        ) -> tuple:
            if not stored_matrices:
                # TODO: REFACTOR WITH no_update
                return (stored_matrices,
                        '',
                        create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors,
                        undone_matrices,
                        output_logs,
                        no_update,
                        no_update,)

            new_stored_matrices = stored_matrices.copy()
            new_stored_vectors = stored_vectors.copy()
            new_previous_vectors = previous_vectors.copy()
            new_undone_matrices = undone_matrices.copy()
            new_output_logs = output_logs

            last_matrix_name = list(new_stored_matrices.keys())[-1]
            last_matrix = np.array(new_stored_matrices[last_matrix_name])
            inverse_matrix = safe_inverse(last_matrix)

            # This is done so that it doesn't delete any new vectors that
            # were made before the undoing.
            new_previous_vectors, output_log_updates = (
                self._handle_newly_added_vectors(
                    new_stored_vectors,
                    new_previous_vectors,
                    inverse_matrix
                ))
            new_output_logs += output_log_updates

            new_undone_matrices[last_matrix_name] = new_stored_matrices.pop(
                last_matrix_name)

            restored_vectors = new_previous_vectors.pop()

            return (new_stored_matrices,
                    str(new_stored_matrices),
                    create_figure(restored_vectors),
                    restored_vectors,
                    new_previous_vectors,
                    new_undone_matrices,
                    new_output_logs)

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            # Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vector-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            Output('animation-interval', 'disabled', allow_duplicate=True),
            Output('animation-steps', 'data', allow_duplicate=True),
            [Input('redo-matrix-button', 'n_clicks'),
             State('matrix-store', 'data'),
             ],
            [State('vector-store', 'data'),
             State('previous-vector-store', 'data'),
             State('undone-matrices-store', 'data'),
             ],
            [State('animation-steps', 'data')
             ],
            prevent_initial_call=True
        )
        def redo_matrix(
                _,
                stored_matrices: MatrixDict,
                stored_vectors: Vectors,
                previous_vectors: list[Vectors],
                undone_matrices: MatrixDict,
                animation_steps: list[Matrix],
        ) -> tuple:
            # A condition to check for an empty `stored_matrices` is
            # not needed because `undone_matrices` may not be empty
            # while `stored_matrices` is empty, but if `undone_matrices`
            # is empty, then `stored_matrices` is for sure empty too.
            # TODO: Refactor with no-update.
            if not undone_matrices:
                return (stored_matrices,
                        str(stored_matrices) if stored_matrices else '',
                        # create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors,
                        undone_matrices,
                        no_update,
                        no_update)

            last_undone_matrix_name = list(undone_matrices.keys())[-1]
            stored_matrices[last_undone_matrix_name] = undone_matrices.pop(
                last_undone_matrix_name)

            previous_vectors.append(stored_vectors.copy())

            most_recent_matrix = np.array(stored_matrices[
                                              last_undone_matrix_name])
            restored_vectors = self.apply_matrix_to_vectors(
                most_recent_matrix,
                stored_vectors
            )

            new_steps = update_animations(animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix)

            return (stored_matrices,
                    str(stored_matrices),
                    # create_figure(restored_vectors),
                    restored_vectors,
                    previous_vectors,
                    undone_matrices,
                    False,
                    new_steps)

        @self.app.callback(
            Output('matrix-store', 'data'),
            Output('matrix-list', 'children'),
            # Output('graph', 'figure'),
            Output('vector-store', 'data'),
            Output('previous-vector-store', 'data'),
            Output('undone-matrices-store', 'data'),
            Output('output-logs', 'children'),
            Output('animation-interval', 'disabled', allow_duplicate=True),
            Output('animation-steps', 'data', allow_duplicate=True),
            [Input('repeat-matrix-button', 'n_clicks'),
             State('repeat-matrix-entry-name', 'value'),
             ],
            [State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vector-store', 'data'),
             State('undone-matrices-store', 'data'),
             State('output-logs', 'children'),
             ],
            [State('animation-steps', 'data')
             ],
            prevent_initial_call=True
        )
        def repeat_matrix(
                _,
                selected_matrix: str | None,
                stored_matrices: MatrixDict,
                stored_vectors: Vectors,
                previous_vectors: list[Vectors],
                undone_matrices: MatrixDict,
                output_logs: str,
                animation_steps: list[Matrix]
        ) -> tuple:
            def _validate_input(
                    name: str | None,
                    stored_matrices_: MatrixDict | tuple | list,
                    output_logs_: str
            ) -> tuple[bool, str]:
                if not stored_matrices_:
                    output_logs_ += 'No matrices exist. '
                    return False, output_logs_
                if name and (name not in stored_matrices_):
                    output_logs_ += f'Matrix "{name}" does not exist. '
                    return False, output_logs_
                return True, output_logs_

            # TODO: Refactor with no-update
            everything_as_they_are = (
                stored_matrices,
                str(stored_matrices),
                # create_figure(stored_vectors),
                stored_vectors,
                previous_vectors,
                undone_matrices,
            )

            valid, new_output_logs = _validate_input(
                name=selected_matrix,
                stored_matrices_=stored_matrices,
                output_logs_=output_logs
            )
            if not valid:
                return everything_as_they_are + (new_output_logs,) + (no_update, no_update)

            if not selected_matrix:
                selected_matrix = list(stored_matrices.keys())[-1]

            new_name = generate_unique_matrix_name(
                selected_matrix,
                stored_matrices
            )

            stored_matrices[new_name] = stored_matrices[selected_matrix]
            previous_vectors.append(stored_vectors.copy())
            new_vectors = self.apply_matrix_to_vectors(
                stored_matrices[selected_matrix],
                stored_vectors
            )

            most_recent_matrix = np.array(list(stored_matrices.values())[-1])
            new_steps = update_animations(animation_steps=animation_steps.copy(), end_matrix=most_recent_matrix)

            return (stored_matrices,
                    str(stored_matrices),
                    # create_figure(new_vectors),
                    new_vectors,
                    previous_vectors,
                    {},
                    output_logs,
                    False,
                    new_steps)

        @self.app.callback(
            Output('add-vector-button', 'children'),
            [Input('new-vector-entry-name', 'value'),
             Input('vector-store', 'data'),
             State('add-vector-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def _change_vector_button_name(
                name_input: str,
                vectors: Vectors,
                n_clicks: int
        ) -> str:
            names = [name for name in vectors]
            if (name_input in names
                    or LOWER_LETTERS[n_clicks % 26] in names):
                return 'Edit Vector'
            else:
                return 'Add Vector'

    def _create_layout(self) -> html.Div:
        return html.Div([
            html.H1('Matrix Visualizer'),
            html.Div([
                dcc.Interval(
                    id='animation-interval',
                    disabled=True,
                    interval=self.interval_ms,
                    n_intervals=0
                ),
                dcc.Store(
                    id='animation-steps',
                    data=[]
                ),
                dcc.Graph(id='graph',
                          figure=create_2d_basis_vectors(self.BASIS_VECTORS),
                          style={'height': '1000%',
                                 'width': '70%',
                                 'display': 'inline-block'}),
                html.Div([
                    html.H2('Vectors'),
                    dcc.Store(id='vector-store', data={**self.BASIS_VECTORS}),
                    html.Div([
                        dcc.Input(
                            id='vector-entry-1',
                            type='number',
                            style={'marginBottom': '10px',
                                   'width': '45%',
                                   'marginRight': '5%'},
                            placeholder='x',
                        ),
                        dcc.Input(
                            id='vector-entry-2',
                            type='number',
                            style={'marginBottom': '10px', 'width': '45%'},
                            size='2',
                            placeholder='y'
                        ),
                    ], style={'width': '100%',
                              'display': 'flex',
                              'flexDirection': 'row',
                              'alignItems': 'center',
                              'justifyContent': 'center'}),
                    html.Div([
                        dcc.Input(
                            id='new-vector-entry-name',
                            type='text',
                            style={'marginRight': '5%',
                                   'marginBottom': '5%',
                                   'width': '20%'},
                            size='2',
                            placeholder='Name'),
                        html.Button(
                            children='Add Vector',
                            id='add-vector-button',
                            style={'marginBottom': '5%',
                                   'width': '70%'},
                            n_clicks=0
                        ),
                    ], style={'width': '100%',
                              'display': 'flex',
                              'flexDirection': 'row',
                              'alignItems': 'center',
                              'justifyContent': 'center'}),
                    html.Div([
                        dcc.Dropdown(
                            id='vector-entry-color',
                            options=[{'label': color.capitalize(),
                                      'value': color}
                                     for color in COLORS],
                            value='black'
                        ),
                        html.Hr(style={'width': '100%',
                                       'marginBottom': '15px',
                                       'marginTop': '15px'}),
                    ]),
                    html.Div([
                        dcc.Input(
                            id='delete-vector-entry-name',
                            type='text',
                            style={'marginRight': '5%',
                                   'marginBottom': '5%',
                                   'width': '20%'},
                            size='2',
                            placeholder='Name'
                        ),
                        html.Button(
                            'Delete Vector',
                            id='delete-vector-button',
                            style={'marginBottom': '5%',
                                   'width': '70%'}
                        ),
                    ])

                ], style={'display': 'flex',
                          'flexDirection': 'column',
                          'marginLeft': '20px',
                          'width': '200px'}),

                html.Div([
                    html.H2('Matrices'),
                    dcc.Store(id='matrix-store', data={}),
                    html.Div([
                        html.Div([
                            dcc.Input(
                                id='matrix-entry-1',
                                type='number',
                                style={'marginBottom': '10px',
                                       'marginRight': '10px',
                                       'width': '80px'},
                                size='2',
                                placeholder='a'
                            ),
                            dcc.Input(
                                id='matrix-entry-2',
                                type='number',
                                style={'marginBottom': '10px',
                                       'width': '80px'},
                                size='2',
                                placeholder='b')
                        ]),
                        html.Div([
                            dcc.Input(
                                id='matrix-entry-3',
                                type='number',
                                style={'marginBottom': '10px',
                                       'marginRight': '10px',
                                       'width': '80px'},
                                size='2',
                                placeholder='c'
                            ),
                            dcc.Input(
                                id='matrix-entry-4',
                                type='number',
                                style={'marginBottom': '10px',
                                       'width': '80px'},
                                size='2',
                                placeholder='d'
                            )
                        ]),
                        html.Div([
                            dcc.Input(
                                id='new-matrix-entry-name',
                                type='text',
                                style={'width': '20%', 'marginRight': '5%'},
                                size='2',
                                placeholder='Name'
                            ),
                            html.Button(
                                'Add Matrix',
                                id='add-matrix-button',
                                n_clicks=0,
                                style={'width': '70%'}
                            ),
                        ], style={'width': '100%',
                                  'display': 'flex',
                                  'flexDirection': 'row',
                                  'alignItems': 'center',
                                  'justifyContent': 'center'}),

                        html.Hr(style={'marginBottom': '15px',
                                       'marginTop': '15px',
                                       'width': '100%'}),

                        html.Div([
                            dcc.Input(
                                id='inverse-matrix-entry-name',
                                type='text',
                                style={'width': '20%', 'marginRight': '5%'},
                                size='2',
                                placeholder='Name'
                            ),
                            html.Button(
                                'Apply Inverse',
                                id='inverse-matrix-button',
                                n_clicks=0,
                                style={'width': '70%'}
                            )
                        ], style={'width': '100%',
                                  'display': 'flex',
                                  'flexDirection': 'row',
                                  'alignItems': 'center',
                                  'justifyContent': 'center'}),

                        html.Hr(style={'marginBottom': '15px',
                                       'marginTop': '15px',
                                       'width': '100%'}),

                        html.Div([
                            dcc.Store(
                                id='previous-vector-store',
                                data=[]
                            ),
                            html.Button(
                                'Undo Last Matrix',
                                id='undo-matrix-button',
                                n_clicks=0,
                                style={'width': '100%', 'marginBottom': '5%'},
                            ),
                            dcc.Store(id='undone-matrices-store', data={}),
                            html.Button(
                                'Redo Last Matrix',
                                id='redo-matrix-button',
                                n_clicks=0,
                                style={'width': '100%'},
                            ),

                            html.Hr(style={'marginBottom': '15px',
                                           'marginTop': '15px',
                                           'width': '100%'}),
                            html.Div([
                                dcc.Input(
                                    id='repeat-matrix-entry-name',
                                    type='text',
                                    style={'width': '20%',
                                           'marginRight': '5%'},
                                    size='2',
                                    placeholder='Name',
                                ),
                                html.Button(
                                    'Repeat Matrix',
                                    id='repeat-matrix-button',
                                    n_clicks=0,
                                    style={'width': '70%'},
                                )
                            ], style={'width': '100%',
                                      'display': 'flex',
                                      'flexDirection': 'row',
                                      'alignItems': 'center',
                                      'justifyContent': 'center'}),
                        ], style={'width': '100%',
                                  'display': 'flex',
                                  'flexDirection': 'column',
                                  'alignItems': 'center',
                                  'justifyContent': 'center'}),

                    ], style={'display': 'flex',
                              'flexDirection': 'column',
                              'alignItems': 'center',
                              'justifyContent': 'center'}),
                ], style={'display': 'flex',
                          'flexDirection': 'column',
                          'alignItems': 'center',
                          'marginLeft': '20px',
                          'width': '200px'})

            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                html.Label('List of Matrices', style={'marginBottom': '10px'}),
                html.Label('', id='matrix-list'),
                *([html.Br()] * 4),
                html.Label('Recent Logs:', style={'marginBottom': '10px'}),
                html.Label('', id='output-logs'),
            ], style={'display': 'flex',
                      'flexDirection': 'column',
                      'height': '500px'})
        ])

    @staticmethod
    def apply_matrix_to_vectors(
            matrix: Matrix,
            vectors: Vectors
    ) -> Vectors:
        new_vectors = vectors.copy()
        vector_list = [np.array([x, y])
                       for _, ((x, y), _) in new_vectors.items()]
        transformed_vectors = [(matrix @ vector).tolist()
                               for vector in vector_list]

        for (name, (_, color)), t_vector in zip(new_vectors.items(),
                                                transformed_vectors):
            new_vectors[name] = [t_vector, color]

        return new_vectors


def main() -> None:
    app = MatrixTransformationsApp(
        {'i-hat': [(1, 0), 'green'],
         'j-hat': [(0, 1), 'red']}
    )
    app.app.run_server(debug=True)


if __name__ == '__main__':
    main()
