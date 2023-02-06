def convert_torch_dtype(dtype):
    import torch
    table = {
        torch.int32: 'int32',
        torch.int64: 'int64',
        torch.float32: 'float32',
        torch.float64: 'float64'
    }
    if dtype not in table:
        raise NotImplementedError(f'Unsupport torch.Tensor dtype {dtype}')

    return table[dtype]


class TensorSpec(object):
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @classmethod
    def from_tensor(cls, tensor):
        import torch
        assert isinstance(tensor, torch.Tensor)
        return cls(shape=tuple(tensor.shape), dtype=convert_torch_dtype(tensor.dtype))

    def __str__(self):
        return str(self.shape) + ', ' + self.dtype

    def __repr__(self):
        return f'TensorSpec({str(self)})'
