from src.types import Vectors, Number
import numpy as np

def calculate_longest_vector_mag(vectors: Vectors) -> Number:
    magnitude_of_longest_vector: Number = -float("inf")
    for (vector_coords), _ in vectors.values():
        vector_mag = np.linalg.norm(np.array(vector_coords))
        magnitude_of_longest_vector = max(
            magnitude_of_longest_vector, float(vector_mag)
        )
    return magnitude_of_longest_vector

