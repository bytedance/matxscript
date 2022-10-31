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
from matx import pipeline
from typing import Tuple, List

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class MyEchoOp1:
    __slots__: Tuple[matx.NativeObject] = ['c_data']

    def __init__(self) -> None:
        self.c_data = matx.make_native_object("MySimpleNativeDataExample")

    def __call__(self) -> bytes:
        return self.c_data.get_content()


class MyEchoOp2:
    __slots__: Tuple[matx.NativeObject] = ['c_data']

    def __init__(self, file_path: str) -> None:
        self.c_data = matx.make_native_object("MyNativeDataExample", file_path)

    def __call__(self) -> List:
        return self.c_data.get_content()


class TestNativeUserData(unittest.TestCase):

    def setUp(self) -> None:
        self.test_file = SCRIPT_PATH + os.sep + "/../data/test.json"

    def test_runtime_native_data(self):
        nd = matx.make_native_object("MySimpleNativeDataExample")
        print(nd.cls_name)
        print(nd.ud_ref)
        print(hasattr(nd, "get_content"))
        content = nd.get_content()
        print(content)

    def test_runtime_native_data_resource(self):
        nd = matx.make_native_object("MyNativeDataExample", self.test_file)
        print(nd.cls_name)
        print(nd.ud_ref)
        print(hasattr(nd, "get_content"))
        content = nd.get_content()
        print(content)

    def test_script_native_data(self):
        def raw_pipeline():
            op1 = MyEchoOp1()
            op2 = MyEchoOp2(self.test_file)
            res1 = op1()
            res2 = op2()
            return res1, res2

        def trace_pipeline():
            op1 = matx.script(MyEchoOp1)()
            op2 = matx.script(MyEchoOp2)(self.test_file)
            res1 = op1()
            res2 = op2()
            return res1, res2

        ans1, ans2 = raw_pipeline()
        res1, res2 = trace_pipeline()
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))

        jit_module = matx.pipeline.Trace(trace_pipeline)
        res1, res2 = jit_module.run(feed_dict={})
        self.assertEqual((res1, res2), (ans1, ans2))

    def test_runtime_native_function(self):
        def native_function_call(a: str, b: int) -> str:
            r = matx.make_native_function("MyNativeFunctionExample")(a.encode(), b)
            return r.decode()

        native_function_call_op = matx.script(native_function_call)
        self.assertEqual("hello: abc, 1", native_function_call("abc", 1))
        self.assertEqual("hello: abc, 1", native_function_call_op("abc", 1))

    def test_native_module_function(self):
        def native_module_func(x: str) -> str:
            return matx.native.MyNativeFunctionExample(x.encode(), 0).decode()

        def generic_native_module_func(x: str) -> str:
            return matx.call_native_function(b'MyNativeFunctionExample', x.encode(), 0).decode()

        op = matx.script(native_module_func)
        r = native_module_func("a")
        op_r = op("a")
        self.assertEqual("hello: a, 0", r)
        self.assertEqual("hello: a, 0", op_r)

        generic_op = matx.script(generic_native_module_func)
        op_r = generic_op("a")
        self.assertEqual("hello: a, 0", r)
        self.assertEqual("hello: a, 0", op_r)

    def test_native_module_object(self):
        def native_module_object() -> str:
            obj = matx.native.MySimpleNativeDataExample()
            return obj.get_content().decode()

        op = matx.script(native_module_object)
        r = native_module_object()
        op_r = op()
        self.assertEqual("hello", r)
        self.assertEqual("hello", op_r)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
