import numpy as np
from dataclasses import dataclass

__all__ = ['Vectors', 'Matrix', 'MatrixDict', 'Number', 'Vector']

type Number = int | float
type Vectors = dict[str, Vector]
type Matrix = np.ndarray
type MatrixDict = dict[str, Matrix]

@dataclass
class Vector:
    """Describes the coordinates and color of one vector."""
    coords: tuple[Number, Number]
    color: str
