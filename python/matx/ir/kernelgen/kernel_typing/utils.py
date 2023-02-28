from ..symbol import is_symbol, equals, simplify


def calculate_output_axis(arr1_shape, arr2_shape, axis_idx):
    axis1 = arr1_shape[axis_idx]
    axis2 = arr2_shape[axis_idx]

    if axis1 is None and axis2 is None:
        raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                          f"because both axes at f{axis_idx} are None after broadcasting")
    if axis1 is None and axis2 is not None:
        return axis2
    if axis1 is not None and axis2 is None:
        return axis1

    if all([is_symbol(axis1), is_symbol(axis2)]):
        if equals(axis1, axis2):
            return simplify(axis1)
        else:
            SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                        f"because {axis1} is not equal to {axis2}.")

    if any([is_symbol(axis1), is_symbol(axis2)]):
        if is_symbol(axis1):
            raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                              f"because {axis1} is a symbol but {axis2} is not.")
        else:
            raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape}"
                              f" because {axis1} is not a symbol but {axis2} is.")

    if axis1 == axis2:
        return axis1

    raise SyntaxError("shapes do not match")


def broadcast(arr1_shape, arr2_shape):
    arr1_shape = list(arr1_shape)
    arr2_shape = list(arr2_shape)
    arr1_shape.reverse()
    arr2_shape.reverse()
    max_ndim = max(len(arr1_shape), len(arr2_shape))
    for i in range(max_ndim):
        if i >= len(arr1_shape):
            arr1_shape.append(None)
        if i >= len(arr2_shape):
            arr2_shape.append(None)

    output_shape = []
    for i in range(max_ndim):
        output_shape.append(calculate_output_axis(arr1_shape, arr2_shape, i))

    output_shape.reverse()
    arr1_shape.reverse()
    arr2_shape.reverse()
    return output_shape, arr1_shape, arr2_shape
