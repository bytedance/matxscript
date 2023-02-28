from .kernel_type import *

PYTYPE_TO_KERNEL_TYPE = {
    bool: ScalarType(bool),
    int: ScalarType(int),
    float: ScalarType(float),
    np.bool_: bool_,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.intc: int32,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.uintc: uint32,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    np.longlong: int64,
    np.ulonglong: uint64
}


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
        aix1 = arr1_shape[i]
        aix2 = arr2_shape[i]
        if aix1 is None and aix2 is None:
            raise SyntaxError("error occurred during broadcasting")
        if aix1 == aix2:
            output_shape.append(aix1)
            continue
        if aix1 is None and aix2 is not None:
            output_shape.append(aix2)
        elif aix1 is not None and aix2 is None:
            output_shape.append(aix1)
        else:
            raise SyntaxError("shapes do not match")

    output_shape.reverse()
    arr1_shape.reverse()
    arr2_shape.reverse()
    return output_shape, arr1_shape, arr2_shape
