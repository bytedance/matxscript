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


class TestKeywordArgs(unittest.TestCase):

    def test_func_explicit_kwargs(self):
        def func(a: int, b: int) -> int:
            return a + b

        def test_call1() -> int:
            return func(3, b=1)

        py_ret = test_call1()
        tx_ret = matx.script(test_call1)()
        self.assertEqual(py_ret, tx_ret)

        def test_call2() -> int:
            return func(a=3, b=1)

        py_ret = test_call2()
        tx_ret = matx.script(test_call2)()
        self.assertEqual(py_ret, tx_ret)

    def test_func_explicit_kwargs_with_default(self):
        def func(x: int, a: int = 5, b: int = 6) -> int:
            return x + a + b

        def test_call() -> Any:
            result = []
            # pass all args
            result.append(func(1, 5, 6))
            result.append(func(1, 5, b=6))
            result.append(func(1, a=5, b=6))
            result.append(func(x=1, a=5, b=6))

            # skip x
            # result.append(func(a=5, b=6))  # error

            # skip a
            result.append(func(1, b=6))
            result.append(func(x=1, b=6))

            # skip b
            result.append(func(1, 5))
            result.append(func(1, a=5))
            result.append(func(x=1, a=5))

            # skip a b
            result.append(func(1))
            result.append(func(x=1))

            return result

        py_ret = test_call()
        tx_ret = matx.script(test_call)()
        self.assertEqual(py_ret, tx_ret)

    def test_func_implicit_kwargs(self):
        def func(a: int, b: int) -> int:
            return a + b

        def test_call1() -> int:
            f: Any = func
            return f(3, b=1)

        py_ret = test_call1()
        tx_ret = matx.script(test_call1)()
        self.assertEqual(py_ret, tx_ret)

        def test_call2() -> int:
            f: Any = func
            return f(a=3, b=1)

        py_ret = test_call2()
        tx_ret = matx.script(test_call2)()
        self.assertEqual(py_ret, tx_ret)

    def test_class_init_explicit_kwargs(self):
        class MyFoo:
            def __init__(self, a: int, b: int):
                self.a: int = a
                self.b: int = b

            def foo(self) -> int:
                return self.a + self.b

        def test_call1() -> int:
            return MyFoo(3, b=1).foo()

        py_ret = test_call1()
        tx_ret = matx.script(test_call1)()
        self.assertEqual(py_ret, tx_ret)

        def test_call2() -> int:
            return MyFoo(a=3, b=1).foo()

        py_ret = test_call2()
        tx_ret = matx.script(test_call2)()
        self.assertEqual(py_ret, tx_ret)

    def test_class_init_explicit_kwargs_with_defaults(self):
        class MyData:
            def __init__(self, x: int, a: int = 5, b: int = 6) -> None:
                self.result: int = x + a + b

        class MyFoo:
            def __init__(self) -> None:
                pass

            def foo(self, x: int, a: int = 5, b: int = 6) -> int:
                return x + a + b

        def test_call_init() -> Any:
            result = []
            # pass all args
            result.append(MyData(1, 5, 6).result)
            result.append(MyData(1, 5, b=6).result)
            result.append(MyData(1, a=5, b=6).result)
            result.append(MyData(x=1, a=5, b=6).result)

            # skip x
            # result.append(func(a=5, b=6))  # error

            # skip a
            result.append(MyData(1, b=6).result)
            result.append(MyData(x=1, b=6).result)

            # skip b
            result.append(MyData(1, 5).result)
            result.append(MyData(1, a=5).result)
            result.append(MyData(x=1, a=5).result)

            # skip a b
            result.append(MyData(1).result)
            result.append(MyData(x=1).result)

            return result

        py_ret = test_call_init()
        tx_ret = matx.script(test_call_init)()
        self.assertEqual(py_ret, tx_ret)

        def test_call_methods() -> Any:
            result = []
            # pass all args
            result.append(MyFoo().foo(1, 5, 6))
            result.append(MyFoo().foo(1, 5, b=6))
            result.append(MyFoo().foo(1, a=5, b=6))
            result.append(MyFoo().foo(x=1, a=5, b=6))

            # skip x
            # result.append(func(a=5, b=6))  # error

            # skip a
            result.append(MyFoo().foo(1, b=6))
            result.append(MyFoo().foo(x=1, b=6))

            # skip b
            result.append(MyFoo().foo(1, 5))
            result.append(MyFoo().foo(1, a=5))
            result.append(MyFoo().foo(x=1, a=5))

            # skip a b
            result.append(MyFoo().foo(1))
            result.append(MyFoo().foo(x=1))

            return result

        py_ret = test_call_methods()
        tx_ret = matx.script(test_call_methods)()
        self.assertEqual(py_ret, tx_ret)

    def test_class_attr_explicit_kwargs(self):
        class MyFoo:
            def __init__(self, a: int, b: int):
                self.a: int = a
                self.b: int = b

            def foo(self, c: int, d: int) -> int:
                return self.a + self.b + c + d

        def test_call1() -> int:
            return MyFoo(3, b=1).foo(3, 1)

        py_ret = test_call1()
        tx_ret = matx.script(test_call1)()
        self.assertEqual(py_ret, tx_ret)

        def test_call2() -> int:
            return MyFoo(a=3, b=1).foo(c=3, d=1)

        py_ret = test_call2()
        tx_ret = matx.script(test_call2)()
        self.assertEqual(py_ret, tx_ret)

    def test_class_attr_implicit_kwargs(self):
        class MyFoo:
            def __init__(self, a: int, b: int):
                self.a: int = a
                self.b: int = b

            def foo(self, c: int, d: int) -> int:
                return self.a + self.b + c + d

        def test_call1() -> int:
            a: Any = MyFoo(3, b=1)
            return a.foo(3, 1)

        py_ret = test_call1()
        tx_ret = matx.script(test_call1)()
        self.assertEqual(py_ret, tx_ret)

        def test_call2() -> int:
            a = MyFoo(a=3, b=1)
            return a.foo(c=3, d=1)

        py_ret = test_call2()
        tx_ret = matx.script(test_call2)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
