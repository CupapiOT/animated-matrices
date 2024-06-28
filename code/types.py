import numpy as np

__all__ = ['Vectors', 'Matrix', 'MatrixDict', 'Number']

type Vectors = dict[str, list[tuple[int, int] | str]]
type Matrix = np.ndarray
type MatrixDict = dict[str, Matrix]
type Number = int | float
