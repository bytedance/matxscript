import numpy as np


class NDArrayType:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.storage = 'cpu'

    def __repr__(self):
        return f'NDArray (dtype={self.dtype.name}, shape={self.shape})'


class ScalarType(NDArrayType):

    def __init__(self, dtype):
        super().__init__([1], dtype)

    def __getitem__(self, shape):
        if isinstance(shape, list) or isinstance(shape, tuple):
            if tuple(shape) == [1]:
                return self
            return NDArrayType(tuple(shape), self.dtype)
        if shape == 1:
            return self
        return NDArrayType((shape,), self.dtype)

    def __repr__(self):
        return f'Scalar (dtype={self.dtype.name}, storage={self.storage})'


int8 = ScalarType(np.int8)
int16 = ScalarType(np.int16)
int32 = ScalarType(np.int32)
int64 = ScalarType(np.int64)
uint8 = ScalarType(np.uint8)
uint16 = ScalarType(np.uint16)
uint32 = ScalarType(np.uint32)
uint64 = ScalarType(np.uint64)
float16 = ScalarType(np.float16)
float32 = ScalarType(np.float32)
float64 = ScalarType(np.float64)
bool_ = None  # todo implement bool
