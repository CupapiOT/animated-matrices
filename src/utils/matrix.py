import numpy as np
from src.types import Vectors, Matrix
import re


def generate_unique_matrix_name(name: str, existing_names) -> str:
    def _remove_duplicate_suffix(base_name: str) -> str:
        non_duplicate_name = base_name
        # A name is a duplicate if it ends in ` (<int>)`.
        name_is_duplicate = re.search(r" \((\d+)\)$", base_name) is not None
        if name_is_duplicate:
            non_duplicate_name = base_name[:-4]
        return non_duplicate_name

    def _find_next_num_in_sequence(names):
        existing_duplicates = [name for name in names if name.startswith(new_name)]
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


def safe_inverse(matrix: Matrix) -> Matrix | None:
    """
    Returns the inverse of the matrix if it exists, otherwise returns
    None.

    Parameters:
    matrix (np.ndarray):  The matrix to be inverted.

    Returns:
    np.ndarray or None: The inverse of the matrix, or None if the
    matrix is not invertible.
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def apply_matrix_to_vectors(matrix: Matrix, vectors: Vectors) -> Vectors:
    new_vectors = vectors.copy()
    vector_list = [np.array([x, y]) for ((x, y), _) in new_vectors.values()]
    transformed_vectors = [(matrix @ vector).tolist() for vector in vector_list]

    for (name, (_, color)), t_vector in zip(new_vectors.items(), transformed_vectors):
        new_vectors[name] = [t_vector, color]

    return new_vectors
