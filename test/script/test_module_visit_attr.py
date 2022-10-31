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

from typing import Any
import unittest

import matx
import external_lib
import external_module


class TestModuleVisitAttr(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_make_cls_by_module_attr(self):
        def call_external_module_attr() -> Any:
            a: external_lib.MyMod = external_lib.MyMod()
            return a.foo()

        py_res = call_external_module_attr()
        tx_res = matx.script(call_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_call_module_attr(self):
        def call_external_module_attr() -> Any:
            return external_lib.add(1., 2.0)

        py_res = call_external_module_attr()
        tx_res = matx.script(call_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_call_module_attr_with_another_call(self):
        def call_external_module_attr_with_another_call() -> Any:
            return external_lib.make_str().encode()

        py_res = call_external_module_attr_with_another_call()
        tx_res = matx.script(call_external_module_attr_with_another_call)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_make_cls_by_module_nested_attr(self):
        def call_external_module_attr() -> Any:
            a: external_module.MyMod = external_module.MyMod()
            return a.foo()

        py_res = call_external_module_attr()
        tx_res = matx.script(call_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_make_cls_by_module_nested_attr_v2(self):
        def call_external_module_attr() -> Any:
            a: external_module.my_class.MyMod = external_module.my_class.MyMod()
            return a.foo()

        py_res = call_external_module_attr()
        tx_res = matx.script(call_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_call_module_nested_attr(self):
        def call_external_module_attr() -> Any:
            return external_module.add(1., 2.0)

        py_res = call_external_module_attr()
        tx_res = matx.script(call_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_module_const_attr(self):
        def visit_external_module_attr() -> Any:
            a1 = external_lib.const_none
            a2 = external_lib.const_int
            a3 = external_lib.const_float
            a4 = external_lib.const_bool
            a5 = external_lib.const_str
            return a1, a2, a3, a4, a5

        py_res = visit_external_module_attr()
        tx_res = matx.script(visit_external_module_attr)()
        self.assertAlmostEqual(py_res, tx_res)

    def test_from_package_import_as(self):
        from external_lib import MyMod as Mod

        class MyTest:
            def __init__(self) -> None:
                self.mm: Mod = Mod()

            def __call__(self,) -> int:
                return self.mm.foo()
        py_ret = MyTest()()
        tx_ret = matx.script(MyTest)()()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
