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


import os
import unittest
import matx


class TestTIRBuffer(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_buffer(self):
        m = matx.ir.PrimVar("m", "int32")
        n = matx.ir.PrimVar("n", "int32")
        k = matx.ir.PrimVar("k", "int32")

        Ab = matx.ir.decl_buffer((m, n), "float32")
        Bb = matx.ir.decl_buffer((n, k), "float32")

        assert isinstance(Ab, matx.ir.Buffer)
        assert Ab.dtype == "float32"
        shape = tuple(Ab.shape)
        # TODO: Simplify the code
        assert len(shape) == 2 and shape[0].same_as(m) and shape[1].same_as(n)

    def test_buffer_vload(self):
        m = matx.ir.PrimVar("m", "int32")
        n = matx.ir.PrimVar("n", "int32")
        Ab = matx.ir.decl_buffer((m, n), "float32", elem_offset=100)
        load = Ab.vload([2, 3])
        matx.ir.assert_structural_equal(load.indices, [2, 3])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
