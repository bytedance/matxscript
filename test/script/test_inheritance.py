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
from typing import Tuple, List, Any, Callable
import unittest
import matx


class TestInheritance(unittest.TestCase):

    def test_basic_inheritance(self):
        class MySimpleBase:
            def __init__(self):
                pass

            def foo(self) -> str:
                return "Base"

        class MyChild(MySimpleBase):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self) -> str:
                return self.foo()

        py_ret = MyChild()()
        tx_ret = matx.script(MyChild)()()
        self.assertEqual(py_ret, tx_ret)

    def test_dup_var_name(self):
        class MyBase:
            def __init__(self) -> None:
                self.val: str = "Base"

            def foo(self) -> str:
                return self.val

        class MyChild(MyBase):
            def __init__(self) -> None:
                super().__init__()
                self.val: str = "Child"

            def __call__(self) -> str:
                return self.foo()

        py_ret = MyChild()()
        tx_ret = matx.script(MyChild)()()
        self.assertEqual(py_ret, tx_ret)

    def test_any_dispatch(self):
        class MyBase:
            def __init__(self) -> None:
                self.val: str = 'Base'

            def foo(self) -> str:
                return self.val

        class MyChild(MyBase):
            def __init__(self) -> None:
                super().__init__()
                self.val: str = 'Child'

            def child_func(self) -> str:
                return self.val

        def any_dispatch_call_child_func() -> str:
            c = MyChild()
            a: Any = c
            return a.child_func()

        py_ret = any_dispatch_call_child_func()
        tx_ret = matx.script(any_dispatch_call_child_func)()
        self.assertEqual(py_ret, tx_ret)

        def any_dispatch_call_super_func() -> str:
            c = MyChild()
            a: Any = c
            return a.foo()

        py_ret = any_dispatch_call_super_func()
        tx_ret = matx.script(any_dispatch_call_super_func)()
        self.assertEqual(py_ret, tx_ret)

        def child_cast_to_super_dispatch() -> str:
            c = MyChild()
            a: MyBase = c
            a = c
            return a.foo()

        py_ret = child_cast_to_super_dispatch()
        tx_ret = matx.script(child_cast_to_super_dispatch)()
        self.assertEqual(py_ret, tx_ret)

        def super_cast_to_child_dispatch() -> str:
            # should raise runtime error
            c = MyBase()
            a: MyChild = c
            return a.foo()

        fn = matx.script(super_cast_to_child_dispatch)
        with self.assertRaises(Exception):
            fn()

        def super_cast_to_child_dispatch2() -> str:
            c = MyChild()
            b: MyBase = c
            b.foo()
            d: MyChild = b
            return d.foo()

        py_ret = super_cast_to_child_dispatch2()
        tx_ret = matx.script(super_cast_to_child_dispatch2)()
        self.assertEqual(py_ret, tx_ret)

        def my_interface(b: MyBase) -> str:
            return b.foo()

        def call_func_child_as_super() -> str:
            c = MyChild()
            return my_interface(c)

        py_ret = call_func_child_as_super()
        tx_ret = matx.script(call_func_child_as_super)()
        self.assertEqual(py_ret, tx_ret)

    def test_basic_super(self):
        class MyBase:
            def __init__(self) -> None:
                pass

            def foo(self) -> str:
                return "MyBase"

        class MySon(MyBase):
            def __init__(self) -> None:
                pass

            def __call__(self) -> str:
                return super().foo()

        py_ret = MySon()()
        tx_ret = matx.script(MySon)()()
        self.assertEqual(py_ret, tx_ret)

    def test_super_with_gap(self):
        class MyBase:
            def __init__(self) -> None:
                pass

            def foo(self) -> str:
                return "MyBase"

        class MySon(MyBase):
            def __init__(self) -> None:
                pass

        class MyGrandSon(MySon):
            def __init__(self) -> None:
                pass

            def __call__(self) -> str:
                return super().foo()

        py_ret = MyGrandSon()()
        tx_ret = matx.script(MyGrandSon)()()
        self.assertEqual(py_ret, tx_ret)

    def test_explicit_super(self):
        class MyBase:
            def __init__(self) -> None:
                pass

            def foo(self) -> str:
                return "MyBase"

        class MySon(MyBase):
            def __init__(self) -> None:
                pass

            def foo(self) -> str:
                return "MySon"

        class MyGrandSon(MySon):
            def __init__(self) -> None:
                pass

        def func() -> str:
            gran_son = MyGrandSon()
            return super(MyGrandSon, gran_son).foo() + super(MySon, gran_son).foo()

        def bad_func() -> str:
            son = MySon()
            return super(MyGrandSon, son).foo() + super(MySon, son).foo()

        py_ret = func()
        tx_ret = matx.script(func)()
        self.assertEqual(py_ret, tx_ret)

        # TODO: fixme
        # with self.assertRaises(Exception):
        #     matx.script(bad_func)

    def test_polymorphism(self):
        class MyBase:
            def __init__(self) -> None:
                super().__init__()
                self.val: str = 'MyBase'

            def foo(self) -> str:
                return 'MyBase::foo()'

            def func(self) -> str:
                return self.foo() + ' MyBase::func() = ' + self.val

        class MySon(MyBase):
            def __init__(self) -> None:
                super().__init__()
                self.val: str = 'MySon'

            def foo(self) -> str:
                return 'MySon::foo()'

            def super_func(self) -> str:
                return super().func()

            def func(self) -> str:
                return self.foo() + ' MySon::func() = ' + self.val

        def func() -> str:
            son = MySon()
            return son.func()

        py_ret = func()
        tx_ret = matx.script(func)()
        self.assertEqual(py_ret, tx_ret)

        def explicit_super() -> str:
            son = MySon()
            return super(MySon, son).func()

        py_ret = explicit_super()
        tx_ret = matx.script(explicit_super)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
