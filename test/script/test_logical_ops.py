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
from typing import Tuple


class TestLogicalOperations(unittest.TestCase):

    def test_prim_logical_operate(self):

        @matx.script
        def prim_and(a: bool, b: bool) -> bool:
            return a and b

        @matx.script
        def prim_and_int(a: bool, b: int) -> bool:
            return a and b

        @matx.script
        def prim_and_const_int(a: bool) -> bool:
            return a and 2

        @matx.script
        def prim_or(a: bool, b: bool) -> bool:
            return a or b

        @matx.script
        def prim_not(a: bool) -> bool:
            return not a

        @matx.script
        def prim_logical_operate(a: bool, b: bool, c: bool) -> bool:
            return a or b and c

        @matx.script
        def short_circuit() -> bool:
            a = []
            return len(a) == 0 or (len(a) == 1 and len(a[0]) == 0)

        self.assertEqual(prim_and(True, prim_or(False, True)), True)
        self.assertEqual(prim_and(False, prim_or(False, True)), False)
        self.assertEqual(prim_and_int(True, 2), True)
        self.assertEqual(prim_and_int(False, 2), False)
        self.assertEqual(prim_and_const_int(True), True)
        self.assertEqual(prim_and_const_int(False), False)
        self.assertEqual(prim_logical_operate(True, False, False), True)
        self.assertEqual(prim_not(True), False)
        self.assertEqual(prim_not(False), True)
        self.assertEqual(short_circuit(), True)

    def test_hlo_logical_operate(self):

        @matx.script
        def hlo_and(a: matx.Dict, b: matx.List) -> bool:
            return not a["flag"] and b[0]

        @matx.script
        def hlo_or(a: matx.Dict, b: matx.List) -> bool:
            return not a["flag"] or b[0]

        d = matx.Dict({"flag": True})
        l = matx.List([True])
        self.assertEqual(hlo_and(d, l), False)
        self.assertEqual(hlo_or(d, l), True)

        class HloLogicalOperate:
            __slots__: Tuple[bool] = ['flag']

            def __init__(self, flag: bool) -> None:
                self.flag = flag

            def __call__(self, value: bool, op: str) -> bool:
                if op == "and":
                    return not self.flag and value
                else:
                    return not self.flag or value

        hlo_logical_operate = matx.script(HloLogicalOperate)(True)
        self.assertEqual(hlo_logical_operate(True, "and"), False)
        self.assertEqual(hlo_logical_operate(True, "or"), True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
