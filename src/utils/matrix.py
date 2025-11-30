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


DUPLICATE_MATRIX_REGEX = re.compile(r"_\{\d+\}$")


def remove_duplicate_suffix(name: str) -> tuple[str, str]:
    """
    Returns the deduplicated base name of a matrix and the duplicate 
    suffix.
    """
    deduplicated_match = re.match(r"(.*)(_\{\d+\})", name)
    if deduplicated_match is not None:
        return deduplicated_match.group(1), deduplicated_match.group(2)
    return name, ""


def generate_duplicate_matrix_name(name: str, existing_names: list[str]) -> str:
    if name not in existing_names:
        return name

    deduplicated_name, _ = remove_duplicate_suffix(name)

    existing_duplicates = [
        name for name in existing_names if name.startswith(deduplicated_name)
    ]

    suffix_numbers_list = []
    for duplicate_name in existing_duplicates:
        suffix = re.search(DUPLICATE_MATRIX_REGEX, duplicate_name)
        if suffix:
            suffix_number = re.search(r"\d+", suffix.group())
            # Safe to ignore what is likely user error from incorrectly
            # naming a matrix.
            if suffix_number is None:
                continue
            suffix_numbers_list.append(int(suffix_number.group()))
    suffix_numbers_list.sort()

    breaks_in_sequence = (
        num
        for num, suffix_num in enumerate(suffix_numbers_list, start=2)
        if num != suffix_num
    )
    try:
        first_missing_number = next(
            breaks_in_sequence,
            suffix_numbers_list[-1] + 1,
        )
    except IndexError:
        first_missing_number = 2

    return f"{deduplicated_name}_{{{first_missing_number}}}"
