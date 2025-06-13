from dash import html, dcc


def input_matrix(
    inputs: list[dcc.Input],
    id: str = "",
    className: str = "",
    abcd_id: str = "",
    abcd_className: str = "",
    ab_id: str = "",
    ab_className: str = "",
    cd_id: str = "",
    cd_className: str = "",
) -> html.Div:
    """
    Returns a 2x1 or 2x2 matrix based on the number of inputs given.

    @param inputs:
    A list of dcc.inputs that act as the x1, y1 (or x1, y1, x2, y2) of the matrix.

    @param id:
    A string for the id of the returned div.

    @param className:
    A string containing the classnames for the returned div. If there are multiple
    classes, use spaces as separators.
    """
    inputs_count: int = len(inputs)
    is_inputs_handleable: bool = inputs_count in (2, 4)
    if not is_inputs_handleable:
        raise NotImplementedError(
            "create_input_matrix only handles 2x1 matrices and 2x2 matrices."
        )

    if inputs_count == 2:
        inputs_container = html.Div(
            id=abcd_id,
            className="input-matrix__inputs" + abcd_className,
            children=inputs,
        )
    else:
        inputs_component: tuple = (
            html.Div(
                id=ab_id,
                className="input-matrix__inputs__a-b" + ab_className,
                children=inputs[:2],
            ),
            html.Div(
                id=cd_id,
                className="input-matrix__inputs__c-d" + cd_className,
                children=inputs[2:],
            ),
        )
        inputs_container = html.Div(
            id=abcd_id,
            className="input-matrix__inputs" + abcd_className,
            children=inputs_component,
        )

    matrix_component = html.Div(
        id=id,
        className="input-matrix " + className,
        children=[
            html.Div(className="left-bracket"),
            inputs_container,
            html.Div(className="right-bracket"),
        ],
    )

    return matrix_component
