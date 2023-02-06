# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
