from matx.ir.kernelgen.kernel_typing import *

from numbers import Number
import numpy as np


def get_dtype(t):
    if isinstance(t, NDArrayType):
        return t.dtype
    elif isinstance(t, (Number, np.bool_)):
        return PYTYPE_TO_KERNEL_TYPE[type(t)]
    else:
        raise TypeError(f"Type {type(t)} of argument {t} is not supported")


def get_shape(t):
    if isinstance(t, NDArrayType):
        return t.shape
    elif isinstance(t, (Number, np.bool_)):
        return None
    else:
        raise TypeError("Type {t} of argument {a} is not supported".format(t=type(t), a=t))


def np_result_dtype(nptypes):
    restype = np.result_type(*nptypes)
    if restype.type not in PYTYPE_TO_KERNEL_TYPE.keys():
        for k in PYTYPE_TO_KERNEL_TYPE.keys():
            if k == restype.type:
                return k
    return restype.type
