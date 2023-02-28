from .kernel_type import NDArrayType


def is_scalar(x: NDArrayType):
    return len(x.shape) == 1 and x.shape[0] == 1


def is_scalar_shape(shape):
    return len(shape) == 1 and shape[0] == 1
