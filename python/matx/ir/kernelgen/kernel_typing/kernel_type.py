from ..symbol import is_symbol


class NDArrayType:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.storage = 'cpu'
        self.symbol_list = [axis for axis in shape if is_symbol(axis)]

    def __repr__(self):
        return f'NDArray (dtype={self.dtype.name}, shape={self.shape})'


class ScalarType(NDArrayType):

    def __init__(self, dtype):
        super().__init__([1], dtype)

    def __getitem__(self, shape) -> NDArrayType:
        if isinstance(shape, list) or isinstance(shape, tuple):
            if tuple(shape) == [1]:
                return self
            return NDArrayType(tuple(shape), self.dtype)
        if shape == 1:
            return self
        return NDArrayType((shape,), self.dtype)

    def __repr__(self) -> str:
        return f'Scalar (dtype={self.dtype.name}, storage={self.storage})'
