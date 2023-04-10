# Copyright 2023 ByteDance Ltd. and/or its affiliates.
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

from matx.ir import _ffi_node_api
import matx
import unittest


class TestLinalgStatementPrint(unittest.TestCase):
    # (a+b)(c-d)/e
    def test_basic_arith_op(self):
        a = matx.ir.PrimVar("a", "float32")
        b = matx.ir.PrimVar("b", "float32")
        t1 = matx.ir.PrimAdd(a, b)
        c = matx.ir.PrimVar("c", "float32")
        d = matx.ir.PrimVar("d", "float32")
        t2 = matx.ir.PrimSub(c, d)
        t3 = matx.ir.PrimMul(t1, t2)
        e = matx.ir.PrimVar("e", "float32")
        t4 = matx.ir.PrimDiv(t3, e)
        ib = matx.ir.ir_builder.create()
        ib.emit(matx.ir.ReturnStmt(t4))
        prim_func = matx.ir.PrimFunc([], [], ib.get(), matx.ir.PrimType("float32"))
        func_name = "basic_arith_op"
        prim_func = prim_func.with_attr("global_symbol", func_name)
        linalg_statement = _ffi_node_api.as_linalg_text(prim_func).decode()
        expected_statement = "%0 = arith.addf %a, %b : f32\n" + "%1 = arith.subf %c, %d : f32\n" + \
            "%2 = arith.mulf %0, %1 : f32\n" + "%3 = arith.divf %2, %e : f32\n" + "return %3"
        self.assertEqual(expected_statement, linalg_statement)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
