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
from typing import List, Any
import numpy
import matx
from matx import FTList


class TestMatxNDArrayConstructor(unittest.TestCase):
    def test_construct_from_list(self):
        @matx.script
        def construct_from_list() -> matx.NDArray:
            l = [0, 1, 2, 3, 4, 5, 6, 7]
            return matx.NDArray(l, [2, 4], "int32")
        mx_nd = construct_from_list()
        np_nd = numpy.arange(8, dtype=numpy.int32).reshape(2, 4)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

    def test_construct_from_ft_int(self):
        @matx.script
        def construct_from_ft_int(dtype: str) -> matx.NDArray:
            l: FTList[int] = [0, 1, 2, 3, 4, 5, 6, 7]
            return matx.NDArray(l, [2, 4], dtype)
        mx_nd = construct_from_ft_int("int32")
        np_nd = numpy.arange(8, dtype=numpy.int32).reshape(2, 4)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

        mx_nd = construct_from_ft_int("int64")
        np_nd = numpy.arange(8, dtype=numpy.int64).reshape(2, 4)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

    def test_construct_from_ft_double(self):
        @matx.script
        def construct_from_ft_double(dtype: str) -> matx.NDArray:
            l: FTList[float] = [0, 1, 2, 3, 4, 5, 6, 7]
            return matx.NDArray(l, [2, 4], dtype)
        mx_nd = construct_from_ft_double("float32")
        np_nd = numpy.arange(8, dtype=numpy.float32).reshape(2, 4)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

        mx_nd = construct_from_ft_double("float64")
        np_nd = numpy.arange(8, dtype=numpy.float64).reshape(2, 4)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

    def test_construct_from_ft_rt_value(self):
        @matx.script
        def get_ft(stop: int) -> FTList[int]:
            ret: FTList[int] = []
            for i in range(stop):
                ret.append(i)
            return ret

        @matx.script
        def construct_from_rt_value(obj: Any, shape: List, dtype: str) -> matx.NDArray:
            return matx.NDArray(obj, shape, dtype)
        mx_nd = construct_from_rt_value([0, 1, 2, 3], [2, 2], "int64")
        np_nd = numpy.arange(4, dtype=numpy.int64).reshape(2, 2)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

        mx_nd = construct_from_rt_value(get_ft(4), [2, 2], "int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

    def test_construct_python(self):
        @matx.script
        def get_ft(stop: int) -> FTList[int]:
            ret: FTList[int] = []
            for i in range(stop):
                ret.append(i)
            return ret
        mx_nd = matx.NDArray([0, 1, 2, 3], [2, 2], "int64")
        np_nd = numpy.arange(4, dtype=numpy.int64).reshape(2, 2)
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))
        mx_nd = matx.NDArray(get_ft(4), [2, 2], "int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

    def test_construct_kwargs(self):
        def test_ndarray_kwargs() -> matx.NDArray:
            return matx.NDArray([1, 2, 3, 4], [2, 2], dtype="int32", device="cpu:0")
        py_ret = test_ndarray_kwargs()
        tx_ret = matx.script(test_ndarray_kwargs)()
        self.assertTrue(numpy.all(tx_ret.asnumpy() == py_ret.asnumpy()))

    def test_construct_from_scalar(self):
        @matx.script
        def construct_from_scalar(obj: Any, shape: List, dtype: str) -> matx.NDArray:
            return matx.NDArray(obj, shape, dtype)

        mx_nd = construct_from_scalar(1, [2, 2], "int64")
        np_nd = numpy.ones(shape=[2, 2], dtype="int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

        mx_nd = construct_from_scalar(1, [2, 2], "float64")
        np_nd = numpy.ones(shape=[2, 2], dtype="int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy().astype(dtype="int64") == np_nd))

    def test_construct_from_one_element_list(self):
        @matx.script
        def construct_from_one_element_list(obj: Any, shape: List, dtype: str) -> matx.NDArray:
            return matx.NDArray(obj, shape, dtype)

        mx_nd = construct_from_one_element_list([1], [2, 2], "int64")
        np_nd = numpy.ones(shape=[2, 2], dtype="int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy() == np_nd))

        mx_nd = construct_from_one_element_list([1], [2, 2], "float64")
        np_nd = numpy.ones(shape=[2, 2], dtype="int64")
        self.assertTrue(numpy.all(mx_nd.asnumpy().astype(dtype="int64") == np_nd))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
