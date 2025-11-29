import numpy as np
from src.config.constants import UPPER_LETTERS
from src.types import Vectors, Matrix
import re


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

    for (name, (_, color)), transformed_vector in zip(
        new_vectors.items(), transformed_vectors
    ):
        new_vectors[name] = [transformed_vector, color]

    return new_vectors


def generate_new_matrix_name(existing_names: list[str]) -> str:
    available_letters = [
        letter for letter in UPPER_LETTERS if letter not in existing_names
    ]
    return (
        available_letters[0]
        if available_letters
        else generate_duplicate_matrix_name("M", existing_names)
    )


def generate_duplicate_matrix_name(name: str, existing_names: list[str]) -> str:
    # A name is a duplicate if it ends in `_int`.
    duplicate_name_suffix = re.compile(r"_\d+$")

    def _remove_duplicate_suffix(base_name: str) -> str:
        deduplicated_name = base_name
        name_is_duplicate = re.search(duplicate_name_suffix, base_name) is not None
        if name_is_duplicate:
            deduplicated_match = re.match(r"^[^_]+", base_name)
            if deduplicated_match is not None:
                deduplicated_name = deduplicated_match.group()
        return deduplicated_name

    def _find_next_num_in_sequence(deduplicated_name, names):
        existing_duplicates = [
            name for name in names if name.startswith(deduplicated_name)
        ]

        number_list = []
        for duplicate_name in existing_duplicates:
            suffix = re.search(duplicate_name_suffix, duplicate_name)
            if suffix:
                number_list.append(int(suffix.group()[1:]))
        number_list.sort()

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
    index_solution = _find_next_num_in_sequence(new_name, existing_names)
    return new_name + f"_{index_solution}"
