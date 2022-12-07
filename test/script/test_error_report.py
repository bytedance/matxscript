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
from typing import Tuple, Any
import unittest
import matx


class TestErrorReport(unittest.TestCase):
    def batch_assertRaisesRegex(self, expected_exception, expected_regex, funcs, *args, **kwargs):
        for func in funcs:
            self.assertRaisesRegex(expected_exception,
                                   expected_regex,
                                   lambda: matx.script(func), *args, **kwargs)

    def test_ret_type(self):
        def func1() -> str:
            return 1

        def func2() -> str:
            s = 1
            return s

        def func3(s: int) -> str:
            return s

        with self.assertRaises(Exception):
            matx.script(func1)
        with self.assertRaises(Exception):
            matx.script(func2)
        with self.assertRaises(Exception):
            matx.script(func3)

    def test_init_param(self):
        class RSS:
            __slots__: Tuple[int] = ['param']

            def __init__(self, param: int) -> None:
                self.param = param

        scripted_rss = matx.script(RSS)
        self.assertRaisesRegex(TypeError,
                               "missing a required argument: 'param' when calling __init__ of RSS",
                               lambda: scripted_rss())
        self.assertRaisesRegex(TypeError,
                               'too many positional arguments when calling __init__ of RSS',
                               lambda: scripted_rss(1, 2))

    def test_assign_to_elem(self):
        def func1(a: Tuple[int, int]) -> None:
            a[0] = 1

        def func2() -> None:
            a = matx.Tuple(1, 2)
            a[0] = 1

        def func3() -> matx.Dict:
            a = matx.Dict()
            a[0] = 1
            return a

        with self.assertRaises(Exception):
            matx.script(func1)
        with self.assertRaises(Exception):
            matx.script(func2)
        self.assertEqual(matx.script(func3)()[0], 1)

    def test_incomplete_tuple(self):
        def func1(a: Tuple) -> None:
            pass

        def func2() -> Tuple:
            pass

        with self.assertRaises(Exception):
            matx.script(func1)
        with self.assertRaises(Exception):
            matx.script(func2)

    def test_member_access(self):
        class RSS:
            __slots__: Tuple[str] = ['x']

            def __init__(self, x: str) -> None:
                self.a = 'a'

            def func(self) -> int:
                return 1

        class RSS2:
            __slots__: Tuple[str] = ['x']

            def __init__(self, x: str) -> None:
                self.x = 'a'

            def func(self) -> int:
                return 1

        class OP1:
            __slots__: Tuple[RSS2] = ['rss']

            def __init__(self, rss: RSS2) -> None:
                self.rss = rss
                self.rss.a = 'b'

            def __call__(self) -> int:
                return self.rss.func()

        class OP2:
            __slots__: Tuple[RSS2] = ['rss']

            def __init__(self, rss: RSS2) -> None:
                self.rss = rss

            def __call__(self) -> int:
                return self.rss.funcxxxx()

        class OP3:
            def __init__(self) -> None:
                pass

            def func(self) -> int:
                pass

        with self.assertRaises(Exception):
            matx.script(RSS)
        with self.assertRaises(Exception):
            matx.script(OP1)
        with self.assertRaises(Exception):
            matx.script(OP2)
        with self.assertRaises(Exception):
            matx.script(OP3)

    def test_dynamic_type(self):
        def func1(flag: bool) -> None:
            a = 1
            if flag:
                a = 's'
            print(a)

        def func2(flag: bool) -> None:
            a = b's'
            if flag:
                a = 's'
            print(a)

        def func3(s: int, flag: bool) -> None:
            if flag:
                s = '12'
            print(s)

        with self.assertRaises(Exception):
            matx.script(func1)
        with self.assertRaises(Exception):
            matx.script(func2)
        with self.assertRaises(Exception):
            matx.script(func3)

    def test_no_return(self):
        def func():
            pass

        def another_func() -> None:
            func()

        with self.assertRaises(Exception):
            matx.script(func)
        with self.assertRaises(Exception):
            matx.script(another_func)

    def test_container_method(self):
        def func() -> None:
            a = matx.Set()
            a.xxx()

        with self.assertRaises(Exception):
            matx.script(func)

    def test_control_flow(self):
        def func_forget_return() -> int:
            a = 5

        def func_forget_return2() -> int:
            a = 5
            if a == 5:
                return 2
            elif a < 5:
                return 3

        def func_has_return() -> int:
            a = 5
            if a == 5:
                return 2
            elif a < 5:
                return 3
            else:
                return 5

        with self.assertRaises(Exception):
            matx.script(func_forget_return)
        with self.assertRaises(Exception):
            matx.script(func_forget_return2)
        self.assertEqual(matx.script(func_has_return)(), 2)

    def test_callable(self):

        def func_call_none() -> None:
            return None()

        with self.assertRaises(Exception):
            matx.script(func_call_none)

        def func_call_int() -> None:
            a = 0
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_int)

        def func_call_float() -> None:
            a = 0.1
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_float)

        def func_call_bool() -> None:
            a = True
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_bool)

        def func_call_list() -> None:
            a = []
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_list)

        def func_call_dict() -> None:
            a = {}
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_dict)

        def func_call_set() -> None:
            a = set()
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_set)

        def func_call_tuple() -> None:
            a = tuple(1, 2)
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_tuple)

        def func_call_str() -> None:
            a = "abc"
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_str)

        def func_call_bytes() -> None:
            a = b"hello"
            b = a()
            return None

        with self.assertRaises(Exception):
            matx.script(func_call_bytes)

    def test_for(self):
        def zip_two_any(a: Any, b: Any) -> Any:
            return [x for x, j, k in zip(a, b)]

        def zip_three_any(a: Any, b: Any, c: Any) -> Any:
            return [x for x, j in zip(a, b, c)]

        def enumerate_any(a: Any) -> Any:
            return [x for x, j, k in enumerate(a)]

        with self.assertRaises(Exception):
            matx.script(zip_two_any)

        with self.assertRaises(Exception):
            matx.script(zip_three_any)

        with self.assertRaises(Exception):
            matx.script(enumerate_any)

    def test_subscript(self):

        def func_subscript_none() -> None:
            return None[0]

        with self.assertRaises(Exception):
            matx.script(func_subscript_none)

        def func_subscript_int() -> None:
            a = 0
            b = a[0]
            return None

        with self.assertRaises(Exception):
            matx.script(func_subscript_int)

        def func_subscript_float() -> None:
            a = 0.1
            b = a[0]
            return None

        with self.assertRaises(Exception):
            matx.script(func_subscript_float)

        def func_subscript_bool() -> None:
            a = True
            b = a[0]
            return None

        with self.assertRaises(Exception):
            matx.script(func_subscript_bool)

    def test_attribute(self):

        def func_attribute_none() -> None:
            return None[0]

        with self.assertRaises(Exception):
            matx.script(func_attribute_none)

        def func_attribute_int() -> None:
            a = 0
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_int)

        def func_attribute_float() -> None:
            a = 0.1
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_float)

        def func_attribute_bool() -> None:
            a = True
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_bool)

        def func_attribute_tuple() -> None:
            a = tuple()
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_tuple)

        def func_attribute_list() -> None:
            a = list()
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_list)

        def func_attribute_dict() -> None:
            a = dict()
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_dict)

        def func_attribute_set() -> None:
            a = set()
            b = a.c
            return None

        with self.assertRaises(Exception):
            matx.script(func_attribute_set)

    def test_with(self):

        def func() -> None:
            with open("nothing") as f:
                print("in with")

        with self.assertRaises(Exception):
            matx.script(func)()

    def test_class_init(self):

        class Foo:

            def name(self) -> str:
                return "Foo"

        with self.assertRaises(Exception):
            matx.script(Foo)

    def test_class_member_annotate(self):

        class TestClass:

            def __init__(self) -> None:
                self.x = "sss"

            def f(self) -> str:
                return self.x

        with self.assertRaises(Exception):
            matx.script(TestClass)

    def test_closure_error(self):

        def func() -> str:

            def closure() -> str:
                return "hello"

            return closure() + " world"

        with self.assertRaises(Exception):
            matx.script(func)

    def test_autofor_error(self):
        a = [1, 2, 3]
        b = {'1_': 1, '2_': 2, '3_': 3}
        c = {1, 2, 3}

        def autofor_list_append_0(outer: list) -> Tuple[list, list]:
            inner = [4, 5]
            useless_cnt = 0
            for item in outer:
                for item2 in inner:
                    useless_cnt += 1
                inner.append(0)
            outer.append(0)
            return outer, inner

        def autofor_list_append_1(a: list) -> None:
            b = [4, 5]
            for item in a:
                for item2 in b:
                    b.append(0)

        def autofor_list_append_2(a: list) -> None:
            b = [4, 5]
            for item in a:
                for item2 in b:
                    a.append(0)

        def autofor_list_extend(a: list) -> None:
            for item in a:
                a.extend([0])

        def autofor_list_remove(a: list) -> None:
            for item in a:
                a.remove(item)

        def autofor_list_pop(a: list) -> None:
            for item in a:
                a.pop()

        def autofor_list_popindex(a: list) -> None:
            for item in a:
                a.pop(0)

        def autofor_dict_pop(b: dict) -> None:
            for item in b:
                b.pop(item)

        def autofor_set_add(c: set) -> None:
            for item in c:
                c.add(0)

        def autofor_set_update(c: set) -> None:
            for item in c:
                c.update({0})

        def autofor_set_discard(c: set) -> None:
            for item in c:
                c.discard(item)

        def autofor_set_pop(c: set) -> None:
            for item in c:
                c.pop()

        def autofor_set_remove(c: set) -> None:
            for item in c:
                c.remove(item)

        outer, inner = matx.script(autofor_list_append_0)(a)
        self.assertEqual(outer, [1, 2, 3, 0])
        self.assertEqual(inner, [4, 5, 0, 0, 0])

        with self.assertRaises(Exception):
            matx.script(autofor_list_append_1)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_list_append_2)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_list_extend)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_list_remove)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_list_pop)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_list_popindex)(a)

        with self.assertRaises(Exception):
            matx.script(autofor_dict_pop)(b)

        with self.assertRaises(Exception):
            matx.script(autofor_set_add)(c)

        with self.assertRaises(Exception):
            matx.script(autofor_set_update)(c)

        with self.assertRaises(Exception):
            matx.script(autofor_set_discard)(c)

        with self.assertRaises(Exception):
            matx.script(autofor_set_pop)(c)

        with self.assertRaises(Exception):
            matx.script(autofor_set_remove)(c)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
