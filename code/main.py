import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from constants import *
from create_figures import create_2d_basis_vectors, create_figure
from project_types import *
from matrix_utils import safe_inverse


class MatrixTransformationsApp:
    matrix_entries = [
        State('matrix-entry-1', 'value'),
        State('matrix-entry-2', 'value'),
        State('matrix-entry-3', 'value'),
        State('matrix-entry-4', 'value')
    ]

    def __init__(self, basis_vectors):
        self.app = Dash('Matrix Transformations')

        self.BASIS_VECTORS = basis_vectors

        self.app.layout = self._create_layout()
        self._register_callbacks()

    def _handle_newly_added_vectors(
            self,
            stored_vectors: Vectors,
            previous_vectors: list[Vectors],
            inverse_matrix: Matrix | None
    ) -> tuple[list[Vectors], str]:
        """Only used within the undo_matrix function, which is defined
        in `self.register_callback`."""
        new_output_log = ''
        if len(stored_vectors) > len(previous_vectors[-1]):
            new_keys = set(stored_vectors) - (set(previous_vectors[-1]))
            new_vector_dict = {key: stored_vectors[key]
                               for key in new_keys}
            if inverse_matrix is not None:
                inverted_new_vectors = self.apply_matrix_to_vectors(
                    inverse_matrix,
                    new_vector_dict
                )
                previous_vectors[-1].update(inverted_new_vectors)
            else:
                previous_vectors[-1].update(new_vector_dict)
                new_output_log = (
                    f'Newly added vector(s) {list(new_keys)} were not '
                    f'changed due to how the previous matrix was unable '
                    f'to be inverted. '
                )

        return previous_vectors, new_output_log

    def _handle_unupdated_vectors(
            self,
            stored_vectors: Vectors,
            previous_vectors: list[Vectors],
            inverse_matrix: Matrix | None
    ) -> tuple[list[Vectors], str]:
        """Only used within the undo_matrix function, which is defined
        in `self.register_callback`."""
        new_output_log = ''
        for key, vector in stored_vectors.items():
            if vector == previous_vectors[-1][key]:
                continue
            if inverse_matrix is not None:
                inverted_edited_vector = (
                    self.apply_matrix_to_vectors(
                        inverse_matrix,
                        {key: vector}
                    ))
                previous_vectors[-1].update(inverted_edited_vector)
            else:
                previous_vectors[-1][key] = vector
                new_output_log = (
                    f'Newly edited vector ({key}) '
                    f'was unable to be properly shown before '
                    f'the undone matrix was applied due to '
                    f'the undone matrix having no inverse. '
                )

        return previous_vectors, new_output_log

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            [Input('add-vector-button', 'n_clicks'),
             State('vector-entry-1', 'value'),
             State('vector-entry-2', 'value'),
             State('vector-entry-color', 'value'),
             State('vector-store', 'data'),
             State('new-vector-entry-name', 'value')
             ],
            prevent_initial_call=True
        )
        def add_or_edit_vector(
                n_clicks: int,
                x_val: Number,
                y_val: Number,
                color: str,
                stored_vectors: Vectors,
                name: str
        ) -> tuple:
            x, y = self._vector_getter(x_val, y_val)
            vector_name = name if name else (LOWER_LETTERS[n_clicks % 26 - 1])
            stored_vectors[vector_name] = [(x, y), color]

            return (create_figure(stored_vectors),
                    stored_vectors)

        @self.app.callback(
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vectors-store', 'data', allow_duplicate=True),
            [Input('delete-vector-button', 'n_clicks'),
             State('delete-vector-entry-name', 'value'),
             State('vector-store', 'data'),
             State('previous-vectors-store', 'data'),
             ],
            prevent_initial_call=True
        )
        def delete_vector(
                _,
                name: str,
                stored_vectors: Vectors,
                old_stored_vectors: list[Vectors]
        ) -> tuple:
            if not name:
                name = list(stored_vectors.keys())[-1]
            if name not in stored_vectors:
                return create_figure(stored_vectors), stored_vectors

            del stored_vectors[name]
            old_stored_vectors.pop()

            return (create_figure(stored_vectors),
                    stored_vectors,
                    old_stored_vectors)

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vectors-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            [Input('add-matrix-button', 'n_clicks'),
             *self.matrix_entries,
             State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vectors-store', 'data'),
             State('new-matrix-entry-name', 'value'),
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
        ) -> tuple:
            a, b, c, d = self._set_empty_matrix_inputs_to_zero(a, b, c, d)

            matrix = np.array([[a, b], [c, d]])
            matrix_name = name if name else UPPER_LETTERS[n_clicks % 26 - 1]

            stored_matrices[matrix_name] = matrix.tolist()
            matrix_list = str({f'{name}': mat
                               for name, mat in stored_matrices.items()})

            previous_vectors.append(stored_vectors.copy())

            most_recent_matrix = np.array(list(stored_matrices.values())[-1])
            new_vectors = self.apply_matrix_to_vectors(
                most_recent_matrix,
                stored_vectors
            )

            return (stored_matrices,
                    matrix_list,
                    create_figure(new_vectors),
                    new_vectors,
                    previous_vectors,
                    {})

        @self.app.callback(
            Output('matrix-store', 'data', allow_duplicate=True),
            Output('matrix-list', 'children', allow_duplicate=True),
            Output('graph', 'figure', allow_duplicate=True),
            Output('vector-store', 'data', allow_duplicate=True),
            Output('previous-vectors-store', 'data', allow_duplicate=True),
            Output('undone-matrices-store', 'data', allow_duplicate=True),
            Output('output-logs', 'children'),
            [Input('undo-matrix-button', 'n_clicks'),
             State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vectors-store', 'data'),
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
                return (stored_matrices,
                        '',
                        create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors,
                        undone_matrices,
                        output_logs)

            new_output_logs = output_logs

            last_matrix_name = list(stored_matrices.keys())[-1]
            last_matrix = stored_matrices[last_matrix_name]
            inverse_matrix = safe_inverse(last_matrix)

            # This is done so that it doesn't delete any new vectors that
            # were made before the undoing.
            previous_vectors, new_output_log = (
                self._handle_newly_added_vectors(
                    stored_vectors,
                    previous_vectors,
                    inverse_matrix
                ))
            new_output_logs += new_output_log

            # This is done so that any recently edited vectors are kept
            # visually consistent after the undo.
            previous_vectors, new_output_log = self._handle_unupdated_vectors(
                stored_vectors,
                previous_vectors,
                inverse_matrix
            )
            new_output_logs += new_output_log

            undone_matrices[last_matrix_name] = stored_matrices.pop(
                last_matrix_name)
            matrix_list = str({f'{name}': mat
                               for name, mat in stored_matrices.items()})

            restored_vectors = previous_vectors.pop()

            return (stored_matrices,
                    matrix_list,
                    create_figure(restored_vectors),
                    restored_vectors,
                    previous_vectors,
                    undone_matrices,
                    new_output_logs)

        @self.app.callback(
            Output('matrix-store', 'data'),
            Output('matrix-list', 'children'),
            Output('graph', 'figure'),
            Output('vector-store', 'data'),
            Output('previous-vectors-store', 'data'),
            Output('undone-matrices-store', 'data'),
            [Input('redo-matrix-button', 'n_clicks'),
             State('matrix-store', 'data'),
             State('vector-store', 'data'),
             State('previous-vectors-store', 'data'),
             State('undone-matrices-store', 'data'),
             ],
            prevent_initial_call=True
        )
        def redo_matrix(
                _,
                stored_matrices: MatrixDict,
                stored_vectors: Vectors,
                previous_vectors: list[Vectors],
                undone_matrices: MatrixDict
        ) -> tuple:
            if (not stored_matrices) and (not undone_matrices):
                return (stored_matrices,
                        '',
                        create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors,
                        undone_matrices)
            if not undone_matrices:
                return (stored_matrices,
                        str({f'{name}': mat
                             for name, mat in stored_matrices.items()}),
                        create_figure(stored_vectors),
                        stored_vectors,
                        previous_vectors,
                        undone_matrices)

            last_undone_matrix_name = list(undone_matrices.keys())[-1]
            stored_matrices[last_undone_matrix_name] = undone_matrices.pop(
                last_undone_matrix_name)
            matrix_list = str({f'{name}': mat
                               for name, mat in stored_matrices.items()})

            previous_vectors.append(stored_vectors.copy())

            most_recent_matrix = np.array(stored_matrices[
                                              last_undone_matrix_name])
            restored_vectors = self.apply_matrix_to_vectors(
                most_recent_matrix,
                stored_vectors
            )

            return (stored_matrices,
                    matrix_list,
                    create_figure(restored_vectors),
                    restored_vectors,
                    previous_vectors,
                    undone_matrices)

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
                                id='previous-vectors-store',
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
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Label('Recent Logs:', style={'marginBottom': '10px'}),
                html.Label('', id='output-logs'),
            ], style={'display': 'flex',
                      'flexDirection': 'column',
                      'height': '500px'})
        ])

    def _vector_getter(
            self,
            x_val: Number,
            y_val: Number
    ) -> tuple[Number, Number]:
        try:
            x = float(x_val)
            y = float(y_val)
        except (ValueError, TypeError):
            return self.BASIS_VECTORS
        return x, y

    @staticmethod
    def apply_matrix_to_vectors(
            matrix: Matrix,
            vectors: Vectors
    ) -> Vectors:
        vector_list = [([x, y])
                       for _, ((x, y), _) in vectors.items()]
        transformed_vectors = [(matrix @ vector).tolist()
                               for vector in vector_list]

        for (name, (_, color)), t_vector in zip(vectors.items(),
                                                transformed_vectors):
            vectors[name] = [t_vector, color]

        return vectors

    @staticmethod
    def _set_empty_matrix_inputs_to_zero(
            a: Number,
            b: Number,
            c: Number,
            d: Number
    ) -> tuple[Number, Number, Number, Number]:
        a = 0 if not a else a
        b = 0 if not b else b
        c = 0 if not c else c
        d = 0 if not d else d
        return a, b, c, d


def main() -> None:
    app = MatrixTransformationsApp(
        {'i-hat': [(1, 0), 'green'],
         'j-hat': [(0, 1), 'red']}
    )
    app.app.run_server(debug=True)


if __name__ == '__main__':
    main()
