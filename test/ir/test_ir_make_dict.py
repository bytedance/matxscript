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


class TestIRMakeDict(unittest.TestCase):

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
        ir_m = _ir.IRModule()
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

    def test_make_return_dict(self):
        def make_return_dict():
            """
            Source:
            def return_dict():
                result = matx.Dict()
                result["hello"] = "world"
                result["hi"] = 32
                return result
            """
            alloca_result = _ir.AllocaVarStmt("result", _ir.DictType())

            ib = _ir.ir_builder.create()
            ib.emit(alloca_result)
            ib.emit_expr(_ir.op.object_set_item(_ir.base.Span(), alloca_result.var,
                                                _ir.StringImm("hello"),
                                                _ir.StringImm("world")))
            ib.emit_expr(_ir.op.object_set_item(_ir.base.Span(), alloca_result.var,
                                                _ir.StringImm("hi"),
                                                _ir.const(32, "int64")))
            ib.emit(_ir.ReturnStmt(alloca_result.var))
            return _ir.Function([], [], ib.get(), _ir.DictType())

        c_func = self.compile_func(make_return_dict)
        # print(self.current_ir)
        # print(self.current_source)
        result = c_func()
        self.assertEqual(result, matx.Dict({b"hello": b"world", b"hi": 32}))

    def test_make_nested_dict(self):
        def make_nested_dict():
            """
            Source:
            def return_list():
                result = matx.Dict()
                item_0 = matx.List()
                item_1 = matx.List()
                item_0.append("hello")
                item_0.append(125)
                item_1.append("hi")
                item_1.append(32)
                result["item0"] = item_0
                result["item1"] = item_1
                return result
            """
            alloca_result = _ir.AllocaVarStmt("result", _ir.DictType())
            alloca_item0 = _ir.AllocaVarStmt("item0", _ir.ListType())
            alloca_item1 = _ir.AllocaVarStmt("item1", _ir.ListType())

            ib = _ir.ir_builder.create()
            ib.emit(alloca_result)
            ib.emit(alloca_item0)
            ib.emit(alloca_item1)

            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item0.var,
                    _ir.StringImm("hello")))
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item0.var,
                    _ir.const(
                        125,
                        "int64")))

            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item1.var,
                    _ir.StringImm("hi")))
            ib.emit_expr(
                _ir.op.object_append(
                    _ir.base.Span(),
                    alloca_item1.var,
                    _ir.const(
                        32,
                        "int64")))

            ib.emit_expr(_ir.op.object_set_item(_ir.base.Span(), alloca_result.var,
                                                _ir.StringImm("item0"),
                                                alloca_item0.var))
            ib.emit_expr(_ir.op.object_set_item(_ir.base.Span(), alloca_result.var,
                                                _ir.StringImm("item1"),
                                                alloca_item1.var))
            ib.emit(_ir.ReturnStmt(alloca_result.var))
            return _ir.Function([], [], ib.get(), _ir.DictType())

        c_func = self.compile_func(make_nested_dict)
        # print(self.current_ir)
        # print(self.current_source)
        result = c_func()
        expect = matx.Dict({b"item0": matx.List([b"hello", 125]), b"item1": matx.List([b"hi", 32])})
        self.assertEqual(result, expect)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
