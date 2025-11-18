from numpy.linalg import inv, LinAlgError
from src.types import Matrix

__all__ = ["safe_inverse"]


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
        return inv(matrix)
    except LinAlgError:
        return None


if __name__ == "__main__":

    def main() -> None:
        from numpy import array

        mat = array([[1, 0], [0, 1]])
        inverted_mat = safe_inverse(mat)
        print(inverted_mat is not None)

    main()
