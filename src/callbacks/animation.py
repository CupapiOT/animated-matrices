from dash import no_update
from dash.dependencies import Input, Output, State
from src.utils.matrix import apply_matrix_to_vectors
from src.utils.vector import calculate_longest_vector_mag
from src.types import MatrixDict, Vectors, Matrix
from src.graph_functions.create_figures import create_figure

def register_animation_callbacks(app_instance):
    @app_instance.app.callback(
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
            last_frame_vectors = apply_matrix_to_vectors(
                last_undone_matrix, stored_vectors
            )
        else:
            try:
                vectors_to_animate = previous_vectors[-1]
            except IndexError:
                vectors_to_animate = app_instance.basis_vectors
            last_frame_vectors = stored_vectors

        current_frame = animation_steps[0]
        interpolated_vectors = apply_matrix_to_vectors(
            current_frame, vectors_to_animate
        )

        # Get the appropriate scale of the graph.
        first_frame_mag = calculate_longest_vector_mag(vectors_to_animate)
        last_frame_mag = calculate_longest_vector_mag(last_frame_vectors)
        graph_scale = max(first_frame_mag, last_frame_mag) * 1.1

        return (
            create_figure(vectors=interpolated_vectors, scale=graph_scale),
            no_update,
            no_update,
            animation_steps[1:] if animation_steps else [],
        )

