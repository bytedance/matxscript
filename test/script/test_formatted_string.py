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
from typing import Dict, List, Any


class TestFormattedString(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_formatted_string_int(self):
        def matx_formatted_string_single_int(i: int) -> str:
            return f"I got {i} !"

        a: int = 5
        result: str = matx.script(matx_formatted_string_single_int)(a)
        expect: str = "I got 5 !"
        self.assertEqual(result, expect)

    def test_formatted_string_multiple_int(self):
        def matx_formatted_string_multiple_int(a: int, b: int, c: int) -> str:
            return f"I got {a} and {b} and {c} !"

        a: int = 5
        b: int = 9
        c: int = 100
        result: str = matx.script(matx_formatted_string_multiple_int)(a, b, c)
        expect: str = "I got 5 and 9 and 100 !"
        self.assertEqual(result, expect)

    def test_formatted_string_mixed_args(self):
        def matx_formatted_string_mixed(a: int, b: float, c: str) -> str:
            return f"I got {a} and {b} and {c} !"

        a: int = 5
        b: float = 0.1
        c: str = "nothing "
        result: str = matx.script(matx_formatted_string_mixed)(a, b, c)
        expect: str = f"I got {a} and {b} and {c} !"
        self.assertEqual(result, expect)

    def test_formatted_string_with_op(self):
        def matx_formatted_string_op(a: int, b: int, c: int) -> str:
            return f"{a}*{b} * {c} =  {a * b * c} !"

        a: int = 5
        b: int = 9
        c: int = 100
        result: str = matx.script(matx_formatted_string_op)(a, b, c)
        expect: str = f"{a}*{b} * {c} =  {a * b * c} !"
        self.assertEqual(result, expect)

    def test_formatted_string_customized_class_str(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

            def __str__(self) -> str:
                return f"self.value = {self.value}"

            def __repr__(self) -> str:
                return "wrong"

        def matx_formatted_string_with_customized_class(a: Any) -> str:
            return f"this obj has {a} !"

        a: myTest = matx.script(myTest)(10)
        b: myTest = myTest(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a)
        expect: str = f"this obj has {b} !"
        self.assertEqual(result, expect)

    def test_formatted_string_mixed_with_class(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

            def __str__(self) -> str:
                return "this is a"

            def __repr__(self) -> str:
                return "wrong"

        def matx_formatted_string_with_customized_class(a: myTest, s: str, i: int, f: float) -> str:
            return f"I got string {s}, int {i}, float {f}, and a obj {a}"

        a: myTest = matx.script(myTest)(10)
        s: str = "just a string"
        i: int = 6
        f: float = 3.14
        b: myTest = myTest(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a, s, i, f)
        expect: str = f"I got string {s}, int {i}, float {f}, and a obj {b}"
        self.assertEqual(result, expect)

    def test_formatted_string_customized_class_repr(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

            def __repr__(self) -> str:
                return f"self.value = {self.value}"

        def matx_formatted_string_with_customized_class(a: Any) -> str:
            return f"this obj has {a} !"

        a: myTest = matx.script(myTest)(10)
        b: myTest = myTest(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a)
        expect: str = f"this obj has {b} !"
        self.assertEqual(result, expect)

    def test_formatted_string_mixed_with_class_repr(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

            def __repr__(self) -> str:
                return "this is a"

        def matx_formatted_string_with_customized_class(a: myTest, s: str, i: int, f: float) -> str:
            return f"I got string {s}, int {i}, float {f}, and a obj {a}"

        a: myTest = matx.script(myTest)(10)
        s: str = "just a string"
        i: int = 6
        f: float = 3.14
        b: myTest = myTest(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a, s, i, f)
        expect: str = f"I got string {s}, int {i}, float {f}, and a obj {b}"
        self.assertEqual(result, expect)

    def test_formatted_string_mixed_with_class_repr_any(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

            def __repr__(self) -> str:
                return "this is a"

        def matx_formatted_string_with_customized_class(a: Any, s: str, i: int, f: float) -> str:
            return f"I got string {s}, int {i}, float {f}, and a obj {a}"

        a: myTest = matx.script(myTest)(10)
        s: str = "just a string"
        i: int = 6
        f: float = 3.14
        b: myTest = myTest(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a, s, i, f)
        expect: str = f"I got string {s}, int {i}, float {f}, and a obj {b}"
        self.assertEqual(result, expect)

    def test_formatted_string_mixed_with_func(self):
        def k(i: str) -> int:
            return 1

        def matx_formatted_string_with_func(a: Any) -> str:
            return f"{a}"

        a: k = matx.script(k)
        result: str = matx.script(matx_formatted_string_with_func)(a)
        expect_prefix: str = f"<function k at 0x"
        self.assertEqual(result[:len(expect_prefix)], expect_prefix)

    def test_formatted_string_mixed_with_class_no_str(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

        def matx_formatted_string_with_customized_class(a: myTest) -> str:
            return f"{a}"

        a: myTest = matx.script(myTest)(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a)
        expect_prefix: str = f"<myTest object at 0x"
        self.assertEqual(result[:len(expect_prefix)], expect_prefix)

    def test_formatted_string_mixed_with_class_no_str_any(self):
        class myTest:
            def __init__(self, value: int):
                self.value: int = value

        def matx_formatted_string_with_customized_class(a: Any) -> str:
            return f"{a}"

        a: myTest = matx.script(myTest)(10)
        result: str = matx.script(matx_formatted_string_with_customized_class)(a)
        expect_prefix: str = f"<myTest object at 0x"
        self.assertEqual(result[:len(expect_prefix)], expect_prefix)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
