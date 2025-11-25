import numpy as np

__all__ = ["Vectors", "Matrix", "MatrixDict", "Number"]

type Number = int | float

# Vectors look like this:
#    {"name" : [(x, y), "color"]}
# Example:
#    {"i-hat": [(1, 0), "green"], "j-hat": [(0, 1), "red"]}
type Vectors = dict[str, list[tuple[Number, Number] | str]]
type Matrix = np.ndarray
type MatrixDict = dict[str, Matrix]
