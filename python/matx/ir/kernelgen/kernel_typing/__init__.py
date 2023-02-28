from .kernel_type import *
from .utils import *
from .broadcast import *
import numpy as np

int8: ScalarType = ScalarType(np.int8)
int16: ScalarType = ScalarType(np.int16)
int32: ScalarType = ScalarType(np.int32)
int64: ScalarType = ScalarType(np.int64)
uint8: ScalarType = ScalarType(np.uint8)
uint16: ScalarType = ScalarType(np.uint16)
uint32: ScalarType = ScalarType(np.uint32)
uint64: ScalarType = ScalarType(np.uint64)
float16: ScalarType = ScalarType(np.float16)
float32: ScalarType = ScalarType(np.float32)
float64: ScalarType = ScalarType(np.float64)
bool_ = None  # todo implement bool

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
