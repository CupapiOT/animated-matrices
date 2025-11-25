import numpy as np
from dash import Dash, html
import dash_bootstrap_components as dbc
from src.config.animation import AnimationSettings
from src.config.constants import META_TAGS
from src.types import Matrix
import dash_bootstrap_components as dbc
from src.tabs.vector import create_vector_section
from src.tabs.matrix import create_matrix_section
from src.components.logs_panel import create_logs_section
from src.components.graph import create_graph_section
from src.callbacks import register_all_callbacks


class MatrixTransformationsApp:
    def __init__(self, basis_vectors):
        self.app = Dash(
            title="Matrix Transformations",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=META_TAGS,
        )

        self.identity = np.identity(2)
        self.basis_vectors = basis_vectors
        self.animation_settings = AnimationSettings()

        self.app.layout = self._create_layout()
        register_all_callbacks(self)

    def _create_layout(self) -> html.Main:
        vector_tab = dbc.Card(dbc.CardBody(create_vector_section(self)))
        matrix_tab = dbc.Card(dbc.CardBody(create_matrix_section()))
        return html.Main(
            [
                html.H1("2D Matrix Visualizer", className="visually-hidden"),
                create_graph_section(self),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            matrix_tab, label="Matrices", tab_id="matrix-section-tab"
                        ),
                        dbc.Tab(
                            vector_tab, label="Vectors", tab_id="vector-section-tab"
                        ),
                    ],
                    id="section-tabs",
                    active_tab="matrix-section-tab",
                ),
                create_logs_section(),
            ],
        )

    def create_frames(
        self,
        end_matrix: np.ndarray,
        start_matrix: np.ndarray | None = None,
        steps: int | None = None,
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

        if start_matrix is None:
            start_matrix = self.identity
        if steps is None:
            steps = self.animation_settings.frames_count

        # The first frame is also returned for future compatibility for
        # exporting animations.
        return [
            (1 - t) * start_matrix + t * end_matrix
            for t in np.linspace(0, 1, num=steps + 1)
        ]

    def update_animations(
        self,
        animation_steps: list[Matrix],
        end_matrix: np.ndarray,
        start_matrix: np.ndarray | None = None,
        steps: int | None = None,
    ) -> list[Matrix]:
        """Returns animation_steps + new_frames."""
        if start_matrix is None:
            start_matrix = self.identity
        if steps is None:
            steps = self.animation_settings.frames_count
        frames = self.create_frames(
            end_matrix=end_matrix, start_matrix=start_matrix, steps=steps
        )
        new_steps = animation_steps + frames
        return new_steps


def main() -> None:
    app = MatrixTransformationsApp(
        {"i-hat": [(1, 0), "green"], "j-hat": [(0, 1), "red"]}
    )
    app.app.run(debug=True)


if __name__ == "__main__":
    main()
