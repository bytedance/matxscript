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
from typing import Any


class MyClass:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "MyClass"

    def __eq__(self, other: Any) -> bool:
        return True


def my_func1() -> Any:
    creator1 = MyClass
    a1: Any = creator1()
    creator2 = matx.script(MyClass)
    a2 = creator2()
    a3 = matx.script(MyClass)()
    return a1, a2, a3


def my_func2() -> Any:
    my_func1_c = matx.script(my_func1)
    b1 = my_func1_c()
    b2 = matx.script(my_func1)()
    return b1, b2


@matx.script
class MyFoo:
    def __init__(self) -> None:
        pass

    def foo(self) -> str:
        return "foo"


def test_my_foo() -> Any:
    b = MyFoo()
    return b.foo()


class TestNestedScript(unittest.TestCase):

    def test_nested_script(self):
        py_ret = my_func2()
        tx_ret = matx.script(my_func2)()
        self.assertEqual(py_ret, tx_ret)

    def test_nested_script_prebuilt_cls(self):
        py_ret = test_my_foo()
        tx_ret = matx.script(test_my_foo)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
