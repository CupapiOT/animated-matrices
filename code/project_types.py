import numpy as np

__all__ = ['Vectors', 'Matrix', 'MatrixDict', 'Number']

type Number = int | float
type Vectors = dict[str, list[tuple[int, int] | str]]
type Matrix = np.ndarray
type MatrixDict = dict[str, Matrix]
