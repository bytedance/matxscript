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
from string import ascii_lowercase


def my_cmp_func(a: int, b: int) -> bool:
    return a >= b


class TestFunction(unittest.TestCase):

    def test_basic_call(self):
        def add_three(x: int) -> int:
            return x + 3

        @matx.script
        def basic_call(x: int, z: int) -> int:
            y: int = add_three(x)
            return y + z + 1

        self.assertEqual(basic_call(1, 1), 6)

    def test_fibonacci(self):

        @matx.script
        def fibonacci(n: int) -> int:
            if n <= 0:
                return 0
            elif n == 1:
                return 0
            elif n == 2:
                return 1
            else:
                return fibonacci(n - 1) + fibonacci(n - 2)

        self.assertEqual(fibonacci(5), 3)

    def test_multi_args(self):

        def hh(a: int, b: int, c: int) -> int:
            return a + b + c

        @matx.script
        def ff(x: int, z: int) -> int:
            y: int = hh(x, 1, 1)
            return y + z + 1

        self.assertEqual(ff(1, 1), 5)

    def test_dead(self):

        def dead(x: int) -> int:
            return 0

        def also_add_three(x: int) -> int:
            return x + 3

        @matx.script
        def te_dead(x: int, z: int) -> int:
            y: int = also_add_three(x)
            return y + z + 1

        self.assertEqual(te_dead(1, 1), 6)

    def test_call_with_more_types(self):

        def list_append_str(x: str) -> matx.List:
            y = matx.List([])
            y.append(x)
            return y

        @matx.script
        def test_list_append() -> matx.List:
            ss = "hello world"
            tt = list_append_str(ss)
            return tt

        self.assertEqual(test_list_append(), matx.List(["hello world"]))

    def test_more_prim_types(self):

        def a_plus_3(x: float) -> float:
            return x + 3

        @matx.script
        def run_f() -> float:
            a = 4.0
            return a_plus_3(a)

        self.assertAlmostEqual(run_f(), 7.0)

    def test_call_chain(self):
        def common() -> int:
            return 42

        @matx.script
        def foo() -> int:
            return common()

        @matx.script
        def bar() -> int:
            return common()

        print(bar())
        print(foo())

    def test_long_chain(self):
        def common() -> int:
            return 42

        def foo() -> int:
            return common()

        @matx.script
        def bar2() -> int:
            return foo() + common()

        assert bar2() == 84

    def test_void(self):

        @matx.script
        def list_mutation(a: matx.List) -> None:
            a.append(1)
            a.append(2)
            a.append(3)

        x = matx.List([4])
        list_mutation(x)
        assert len(x) == 4

    def test_output_multi_symbol(self):

        @matx.script
        def return_tuple(a: int, b: int) -> Tuple[int, int]:
            return a + b, a * b

        a = 2
        b = 4
        ret_add, ret_mul = return_tuple(a, b)
        self.assertEqual(ret_add, 6)
        self.assertEqual(ret_mul, 8)

    def test_default_args_in_compiling(self):

        def callee(x: int, y: int = 3) -> int:
            return x + y

        @matx.script
        def caller(x: int, y: int) -> int:
            if y == -1:
                return callee(x)
            else:
                return callee(x, y)

        self.assertEqual(caller(2, -1), 5)
        self.assertEqual(caller(2, 4), 6)

    def test_default_args_in_calling(self):

        @matx.script
        def func1(x: int, y: int = 3) -> int:
            return x + y

        @matx.script
        def func2(x: str, y: Any = None) -> str:
            if y is None:
                y = 5
            return x * y

        self.assertEqual(func1(2), 5)
        self.assertEqual(func1(2, -1), 1)
        self.assertEqual(func2('ab', 4), 'abababab')
        self.assertEqual(func2('ab'), 'ababababab')

    def test_default_arg_from_import(self):
        @matx.script
        def func(s: str = ascii_lowercase) -> str:
            return s

        self.assertEqual(func(), ascii_lowercase)

    def test_global_var(self):
        @matx.script
        def func() -> str:
            return ascii_lowercase

        self.assertEqual(func(), ascii_lowercase)

    def test_fun_lambda(self):

        def func() -> None:
            f = lambda x: x * 2
            print(f(4))

        try:
            matx.script(func)()
        except Exception as e:
            assert "Lambda expression" in str(e)

    def test_func_as_object(self):
        # TODO: fix bug
        # def my_cmp_func(a: int, b: int) -> bool:
        #     return a >= b

        class MySortOp:
            def __init__(self) -> None:
                pass

            def bubbleSortV1(self, arr: List, cmp_func: my_cmp_func) -> List:
                n = len(arr)
                for i in range(n):
                    for j in range(0, n - i - 1):
                        if cmp_func(arr[j], arr[j + 1]):
                            tmp = arr[j]
                            arr[j] = arr[j + 1]
                            arr[j + 1] = tmp
                            # arr[j], arr[j + 1] = arr[j + 1], arr[j]
                return arr

            def bubbleSortV2(self, arr: List, cmp_func: Callable) -> List:
                n = len(arr)
                for i in range(n):
                    for j in range(0, n - i - 1):
                        if cmp_func(arr[j], arr[j + 1]):
                            tmp = arr[j]
                            arr[j] = arr[j + 1]
                            arr[j + 1] = tmp
                            # arr[j], arr[j + 1] = arr[j + 1], arr[j]
                return arr

            def __call__(self, arr: List) -> List:
                x = my_cmp_func
                self.bubbleSortV2(arr, x)
                return self.bubbleSortV1(arr, my_cmp_func)

        def raw_pipeline(arr: List):
            sort_op = MySortOp()
            res = sort_op(arr)
            return res

        def script_pipeline(arr: List):
            sort_op = matx.script(MySortOp)()
            res = sort_op(arr)
            return res

        arr = [3, 7, 1, 8]
        ans = raw_pipeline(arr)
        print("runtime: ", ans)
        res = script_pipeline(arr)
        print("compile: ", res)
        self.assertEqual(res, ans)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
