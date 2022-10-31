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
import numpy
import matx


class TestMatxNDArrayFloat16(unittest.TestCase):
    def test_construct_from_list(self):
        def construct_from_list() -> matx.NDArray:
            list_data = [0, 1, 2, 3, 4, 5, 6, 7]
            return matx.NDArray(list_data, [2, 4], "float16")

        py_ret = construct_from_list()
        tx_func = matx.script(construct_from_list)
        tx_ret = tx_func()
        np_nd = numpy.arange(8, dtype=numpy.int32).reshape(2, 4).astype(numpy.float16)
        self.assertTrue(numpy.all(py_ret.asnumpy() == np_nd))
        self.assertTrue(numpy.all(tx_ret.asnumpy() == np_nd))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
