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


class TestDynamicVarType(unittest.TestCase):

    def test_dynamic_generic_var_type(self):
        def dynamic_generic_var_type() -> Any:
            s = "hello"
            s = s.encode()
            return s

        py_ret = dynamic_generic_var_type()
        tx_ret = matx.script(dynamic_generic_var_type)()
        self.assertEqual(py_ret, tx_ret)

    def test_dynamic_loop_var_type(self):
        def dynamic_loop_var_type() -> Any:
            s = "hello"
            li = []
            for si in s:
                si = si.encode()
                li.append(si)
            return b' '.join(li)

        py_ret = dynamic_loop_var_type()
        tx_ret = matx.script(dynamic_loop_var_type)()
        self.assertEqual(py_ret, tx_ret)

    def test_dynamic_arg_type(self):
        def dynamic_arg_type(s: str) -> Any:
            s = s.encode()
            return s

        input_s = "hello"
        py_ret = dynamic_arg_type(input_s)
        tx_ret = matx.script(dynamic_arg_type)(input_s)
        self.assertEqual(py_ret, tx_ret)

    def test_dynamic_var_type_across_scope(self):
        def dynamic_var_across_if(cond: bool) -> Any:
            s = "Hello"
            if cond:
                s = s.encode()
                return s
            else:
                s = len(s.encode())
                return s

        input_s = True
        py_ret = dynamic_var_across_if(input_s)
        tx_ret = matx.script(dynamic_var_across_if)(input_s)
        self.assertEqual(py_ret, tx_ret)

        input_s = False
        py_ret = dynamic_var_across_if(input_s)
        tx_ret = matx.script(dynamic_var_across_if)(input_s)
        self.assertEqual(py_ret, tx_ret)

        def dynamic_var_across_if2(cond: bool) -> Any:
            s = "Hello"
            if cond:
                s = 3
                return s
            return s

        input_s = True
        py_ret = dynamic_var_across_if2(input_s)
        tx_ret = matx.script(dynamic_var_across_if2)(input_s)
        self.assertEqual(py_ret, tx_ret)

        def dynamic_var_across_for_loop() -> Any:
            s = "Hello"
            for i in range(1):
                s = s.encode()
                return s

        py_ret = dynamic_var_across_for_loop()
        tx_ret = matx.script(dynamic_var_across_for_loop)()
        self.assertEqual(py_ret, tx_ret)

        def dynamic_var_across_for_loop2() -> Any:
            s = "Hello"
            for i in range(1):
                s = i
                return s
            return s

        py_ret = dynamic_var_across_for_loop2()
        tx_ret = matx.script(dynamic_var_across_for_loop2)()
        self.assertEqual(py_ret, tx_ret)

        def dynamic_var_across_while_loop() -> Any:
            s = "Hello"
            i = 0
            while i < 1:
                s = s.encode()
                return s

        py_ret = dynamic_var_across_while_loop()
        tx_ret = matx.script(dynamic_var_across_while_loop)()
        self.assertEqual(py_ret, tx_ret)

        def dynamic_var_across_try() -> Any:
            s = "Hello"
            try:
                s = s.encode()
                return s
            except:
                s = len(s.encode())
                return s

        py_ret = dynamic_var_across_try()
        tx_ret = matx.script(dynamic_var_across_try)()
        self.assertEqual(py_ret, tx_ret)

    def test_unexpected_dynamic_var_type(self):
        def unexpected_dynamic_var_across_if1(cond: bool) -> Any:
            s = "Hello"
            if cond:
                s = s.encode()
            return s

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_if1)

        def unexpected_dynamic_var_across_if2(cond1: bool, cond2: bool) -> Any:
            s = "Hello"
            if cond1:
                s = s.encode()
            if cond2:
                a = s

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_if2)

        def unexpected_dynamic_var_across_if3(cond1: bool, cond2: bool) -> Any:
            s = "Hello"
            if cond1:
                s = s.encode()
            if cond2:
                if cond1:
                    a = s

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_if3)

        def unexpected_dynamic_var_across_loop1(loop_num: int) -> Any:
            s = "Hello"
            for i in range(loop_num):
                s = s.encode()
            return s

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_loop1)

        def unexpected_dynamic_var_across_loop2(loop_num: int) -> Any:
            s = "Hello"
            for i in range(loop_num):
                s = s.encode()

            a = 0
            for i in range(loop_num):
                a = a + len(s)

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_loop2)

        def unexpected_dynamic_var_across_loop3(loop_num: int) -> Any:
            s = "Hello"
            for i in range(loop_num):
                s = s.encode()

            a = 0
            for i in range(loop_num):
                while i < 2:
                    a = a + len(s)

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_loop3)

        def unexpected_dynamic_var_across_nested1(loop_num: int, cond1: bool) -> Any:
            s = "Hello"
            for i in range(loop_num):
                s = s.encode()
            if cond1:
                a = len(s)
                print(a)

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_nested1)

        def unexpected_dynamic_var_across_nested2(loop_num: int, cond1: bool) -> Any:
            s = "Hello"
            if cond1:
                s = s.encode()
            a = 0
            for i in range(loop_num):
                a = a + len(s)

        with self.assertRaises(Exception):
            matx.script(unexpected_dynamic_var_across_nested2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
