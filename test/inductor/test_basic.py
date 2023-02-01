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

import unittest
import matx
import torch
import numpy as np


class BasicTests(unittest.TestCase):
    def test_basics(self):
        from matx import toolchain
        toolchain.USE_SO_CACHE = False

        def add_relu(a, b):
            c = a + b
            c = torch.nn.functional.relu(c)
            return c,

        sizes = [(5,), (10,), (2, 3), (4, 5, 6)]
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for size in sizes:
            for dtype in dtypes:
                a_numpy = np.random.randn(*size).astype(dtype)
                b_numpy = np.random.randn(*size).astype(dtype)

                example_inputs = [torch.from_numpy(np.random.randn(*size).astype(dtype)),
                                  torch.from_numpy(np.random.randn(*size).astype(dtype))]

                add_relu_kernel = matx.inductor_script(example_inputs)(add_relu)

                a_tensor = torch.from_numpy(a_numpy)
                b_tensor = torch.from_numpy(b_numpy)

                a_ndarray = matx.NDArray([], a_numpy.shape, str(a_numpy.dtype))
                a_ndarray.from_numpy(a_numpy)
                b_ndarray = matx.NDArray([], b_numpy.shape, str(b_numpy.dtype))
                b_ndarray.from_numpy(b_numpy)

                c_tensor_expected = add_relu(a_tensor, b_tensor)[0]
                c_ndarray: matx.NDArray = add_relu_kernel(a_ndarray, b_ndarray)[0]
                c_tensor = c_ndarray.torch()

                # TODO: there seems a strange cache behavior of JITOp, without the following line,
                # it fails.
                del add_relu_kernel

                torch.testing.assert_close(c_tensor_expected, c_tensor)


if __name__ == '__main__':
    unittest.main()
