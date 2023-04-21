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
    def test_float_arith_op(self):
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
        prim_func = matx.ir.PrimFunc([a, b, c, d, e], [], ib.get(), matx.ir.PrimType("float32"))
        func_name = "basic_arith_op"
        prim_func = prim_func.with_attr("global_symbol", func_name)
        linalg_statement = _ffi_node_api.as_linalg_text(prim_func).decode()
        expected_statement = """
func.func @basic_arith_op(%a: f32, %b: f32, %c: f32, %d: f32, %e: f32)->f32{
%0 = arith.addf %a, %b : f32
%1 = arith.subf %c, %d : f32
%2 = arith.mulf %0, %1 : f32
%3 = arith.divf %2, %e : f32
func.return %3 :f32
}"""
        self.assertEqual(expected_statement.strip(), linalg_statement.strip())

    def test_int_arith_op(self):
        a = matx.ir.PrimVar("a", "int32")
        b = matx.ir.PrimVar("b", "int32")
        t1 = matx.ir.PrimAdd(a, b)
        c = matx.ir.PrimVar("c", "int32")
        d = matx.ir.PrimVar("d", "int32")
        t2 = matx.ir.PrimSub(c, d)
        t3 = matx.ir.PrimMul(t1, t2)
        e = matx.ir.PrimVar("e", "int32")
        t4 = matx.ir.PrimDiv(t3, e)
        ib = matx.ir.ir_builder.create()
        ib.emit(matx.ir.ReturnStmt(t4))
        prim_func = matx.ir.PrimFunc([a, b, c, d, e], [], ib.get(), matx.ir.PrimType("int32"))
        func_name = "basic_arith_op"
        prim_func = prim_func.with_attr("global_symbol", func_name)
        linalg_statement = _ffi_node_api.as_linalg_text(prim_func).decode()
        expected_statement = """
func.func @basic_arith_op(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32)->i32{
%0 = arith.addi %a, %b : i32
%1 = arith.subi %c, %d : i32
%2 = arith.muli %0, %1 : i32
%3 = arith.divi %2, %e : i32
func.return %3 :i32
}"""
        self.assertEqual(expected_statement.strip(), linalg_statement.strip())

    def test_pointer_op(self):
        ptr_type = matx.ir.PointerType(matx.ir.PrimType("int32"))
        a = matx.ir.PrimVar("a", ptr_type)
        ib = matx.ir.ir_builder.create()
        ib.emit(matx.ir.ReturnStmt(a))
        prim_func = matx.ir.PrimFunc([a], [], ib.get(), ptr_type)
        func_name = "basic_arith_op"
        prim_func = prim_func.with_attr("global_symbol", func_name)
        linalg_statement = _ffi_node_api.as_linalg_text(prim_func).decode()
        expected_statement = """
func.func @basic_arith_op(%a: memref<?xi32>)->memref<?xi32>{
func.return %a :memref<?xi32>
} """
        self.assertEqual(expected_statement.strip(), linalg_statement.strip())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
