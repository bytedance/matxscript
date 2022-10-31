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
from typing import List, Dict, Set, Tuple
import unittest
import matx

get_move_var_lineno = matx.get_global_func("ir.MoveOptimizer_GetMoveVarAndLineno")


class TestAutoMoveV2(unittest.TestCase):
    def test_func_match(self):
        def test_normal_move(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = "hello world".split(b)
            return d

        func = matx.ir_module(test_normal_move)["test_normal_move"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0][0], b'a')
        self.assertEqual(info[0][1], 29)  # 29 mean source code line

        def test_if_branch_move(cond: bool) -> List[str]:
            a = "hello"
            li = []
            if cond:
                b = "world"
                li.append(a)
                li.append(b)
            else:
                c = a
                li.append(c)
                li.append(a)
            return li

        func = matx.ir_module(test_if_branch_move)["test_if_branch_move"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0][0], b'b')
        self.assertEqual(info[0][1], 47)
        self.assertEqual(info[1][0], b'c')
        self.assertEqual(info[1][1], 50)

        def test_auto_for_move() -> List[str]:
            a = "hello"
            b = "hi"
            li = []
            for x in a:
                d = "hello"
                li.append(x)
                li.append(d)
                li.append(b)
            return li

        func = matx.ir_module(test_auto_for_move)["test_auto_for_move"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0][0], b'x')
        self.assertEqual(info[0][1], 68)
        self.assertEqual(info[1][0], b'd')
        self.assertEqual(info[1][1], 69)

        def test_for_range_move() -> List[str]:
            a = "hello"
            b = "hi"
            li = []
            for i in range(len(a)):
                d = "hello"
                li.append(d)
                li.append(b)
            return li

        func = matx.ir_module(test_for_range_move)["test_for_range_move"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0][0], b'a')
        self.assertEqual(info[0][1], 85)
        self.assertEqual(info[1][0], b'd')
        self.assertEqual(info[1][1], 87)

        def test_while_move() -> List[str]:
            a = "hello"
            b = "hi"
            li = []
            while len(a):
                d = "hello"
                li.append(d)
                li.append(b)
            return li

        func = matx.ir_module(test_while_move)["test_while_move"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0][0], b'd')
        self.assertEqual(info[0][1], 105)

        def test_try_except() -> List[str]:
            b = "hi"
            li = []
            try:
                d1 = "d1"
                li.append(b)
                li.append(d1)
            except:
                d2 = "d2"
                li.append(b)
                li.append(d2)
            return li

        func = matx.ir_module(test_try_except)["test_try_except"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0][0], b'd1')
        self.assertEqual(info[0][1], 121)
        self.assertEqual(info[1][0], b'd2')
        self.assertEqual(info[1][1], 125)

    def test_func_not_match(self):
        def test_use_twice1() -> None:
            b = "hi"
            c = b + b

        func = matx.ir_module(test_use_twice1)["test_use_twice1"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_twice2() -> None:
            b = "hi"
            c = b + b[0:1]

        func = matx.ir_module(test_use_twice2)["test_use_twice2"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_after_move0() -> None:
            b = "hi"
            c = b
            print(b, b)

        func = matx.ir_module(test_use_after_move0)["test_use_after_move0"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_after_move1() -> None:
            b = "hi"
            c = b
            print(b, b)

        func = matx.ir_module(test_use_after_move1)["test_use_after_move1"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_after_move2(cond: bool) -> None:
            b = "hi"
            c = b
            if cond:
                print(b, b)

        func = matx.ir_module(test_use_after_move2)["test_use_after_move2"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_after_move3(cond: bool) -> None:
            b = "hi"
            c = b
            while cond:
                print(b, b)
                break

        func = matx.ir_module(test_use_after_move3)["test_use_after_move3"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)

        def test_use_after_move4() -> None:
            b = "hi"
            c = b
            for i in range(2):
                print(b, b)

        func = matx.ir_module(test_use_after_move4)["test_use_after_move4"]
        info = get_move_var_lineno(func)
        self.assertEqual(len(info), 0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
