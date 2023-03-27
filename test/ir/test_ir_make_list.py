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
import ctypes
import unittest
import matx
from matx import ir as _ir

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestIRMakeList(unittest.TestCase):

    def setUp(self) -> None:
        self.build_module = matx.get_global_func("module.build.c")
        self.current_ir = ""
        self.current_source = ""
        self.dso_prefix_path = SCRIPT_PATH + os.sep + matx.toolchain.LIB_PATH

    def compile_func(self, func):
        """Compile ir function and return c_func

        Parameters
        ----------
        func : Function
            used to make ir func

        Returns
        -------
        c_func : Function
            compiled function

        """
        func_name = func.__name__
        ir_func = func()
        ir_func = ir_func.with_attr("global_symbol", func_name)
        ir_m = matx.ir.IRModule()
        ir_m.add(ir_func)
        ir_m.set_main(func_name)
        self.current_ir = str(ir_m)
        rt_m = self.build_module(ir_m)
        self.current_source = rt_m.get_source()

        dso_path = "%s/lib_ir_%s.so" % (self.dso_prefix_path, func_name)
        if not os.path.exists(self.dso_prefix_path):
            os.mkdir(self.dso_prefix_path)
        rt_m.export_library(dso_path)

        rt_module = matx.runtime.load_module(dso_path)
        c_func = rt_module.get_function(func_name)
        return c_func

    def test_make_return_list(self):
        def make_return_list():
            """
            Source:
            def return_list():
                result = matx.List()
                result.append("hello")
                result.append(125)
                return result
            """
            alloca_result = matx.ir.AllocaVarStmt("result", matx.ir.ListType())
            ib = matx.ir.ir_builder.create()
            ib.emit(alloca_result)
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_result.var,
                    matx.ir.StringImm("hello")))
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_result.var,
                    matx.ir.const(
                        125,
                        "int64")))
            ib.emit(matx.ir.ReturnStmt(alloca_result.var))
            return matx.ir.Function([], [], ib.get(), matx.ir.ListType())

        c_func = self.compile_func(make_return_list)
        result = c_func()
        # print(self.current_ir)
        # print(self.current_source)
        self.assertEqual(result, matx.List([b"hello", 125]))

    def test_make_nested_list(self):
        def make_nested_list():
            """
            Source:
            def return_list():
                result = matx.List()
                item_0 = matx.List()
                item_1 = matx.List()
                item_0.append("hello")
                item_0.append(125)
                item_1.append("hi")
                item_1.append(32)
                result.append(item_0)
                result.append(item_1)
                return result
            """
            alloca_result = matx.ir.AllocaVarStmt("result", matx.ir.ListType())
            alloca_item0 = matx.ir.AllocaVarStmt("item0", matx.ir.ListType())
            alloca_item1 = matx.ir.AllocaVarStmt("item1", matx.ir.ListType())

            ib = matx.ir.ir_builder.create()
            ib.emit(alloca_result)
            ib.emit(alloca_item0)
            ib.emit(alloca_item1)

            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item0.var,
                    matx.ir.StringImm("hello")))
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item0.var,
                    matx.ir.const(
                        125,
                        "int64")))

            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item1.var,
                    matx.ir.StringImm("hi")))
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item1.var,
                    matx.ir.const(
                        32,
                        "int64")))

            ib.emit_expr(_ir.op.object_append(_ir.base.Span(), alloca_result.var, alloca_item0.var))
            ib.emit_expr(_ir.op.object_append(_ir.base.Span(), alloca_result.var, alloca_item1.var))

            ib.emit(matx.ir.ReturnStmt(alloca_result.var))
            return matx.ir.Function([], [], ib.get(), matx.ir.ListType())

        c_func = self.compile_func(make_nested_list)
        result = c_func()
        # print(self.current_ir)
        # print(self.current_source)
        self.assertEqual(result, matx.List([matx.List([b"hello", 125]), matx.List([b"hi", 32])]))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
