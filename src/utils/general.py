from src.types import Number

def set_nonetype_to_zero(
    *values: Number | None,
) -> list:
    """
    Takes any number of int-type or float-type values and filters them,
    setting each value to 0 if it is a NoneType.

    Parameter:
    *values = Any numbers.

    Returns:
    The same number of values, with each value either staying the same
    or converted to 0 if it was a NoneType.
    """
    return [val if val is not None else 0 for val in values]
