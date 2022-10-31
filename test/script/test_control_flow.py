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
from typing import Any, Dict, List, Set, Tuple
from matx import FTList, FTSet, FTDict


class TestControlFlow(unittest.TestCase):

    def test_basic(self):

        @matx.script
        def naive_ifelse(x: int) -> int:
            if x > 3:
                return x + 1
            else:
                return x - 1

        @matx.script
        def if_expr(x: int) -> int:
            return x + 1 if x > 3 else x - 1

        @matx.script
        def naive_forloop(x: int) -> int:
            for i in range(5):
                if x + i > 3:
                    return x + i
            return 0

        @matx.script
        def sumup(n: int) -> int:
            s = 0
            for i in range(n):
                s += (i + 1)
            return s

        @matx.script
        def assert_true() -> int:
            assert True, "assert successful"
            return 0

        @matx.script
        def assert_false() -> int:
            assert False, "assert failed"
            return -1

        self.assertEqual(naive_ifelse(5), 6)
        self.assertEqual(naive_ifelse(2), 1)
        self.assertEqual(if_expr(5), 6)
        self.assertEqual(if_expr(2), 1)
        self.assertEqual(naive_forloop(2), 4)
        self.assertEqual(sumup(100), 5050)
        self.assertEqual(assert_true(), 0)
        with self.assertRaises(Exception) as context:
            assert_false()

    def test_while(self):

        @matx.script
        def whileloop(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                i += 1
                s += i
            return s

        self.assertEqual(whileloop(10), 55)

    def test_branch(self):

        @matx.script
        def breaktest(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                i += 1
                s += i
                if s > 10:
                    break
            return s

        @matx.script
        def continuetest(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                i += 1
                s += i
                if s <= 10:
                    continue
                else:
                    break
            return s

        @matx.script
        def more_control_flow(a: int) -> int:
            sum_ = 0
            while a > 0:
                sum_ += a
                a = a - 1
                if a == 3 or a == 4 and a > 0:
                    break
                if a == 10:
                    a = a - 2
                    continue
            return sum_

        @matx.script
        def nested_loop(a: int) -> int:
            sum_ = 0
            cnt = 0
            while cnt < 5:
                for i in range(a):
                    sum_ += (i + 1)
                cnt += 1
            return sum_

        self.assertEqual(breaktest(8), 15)
        self.assertEqual(continuetest(8), 15)
        self.assertEqual(more_control_flow(10), 45)
        self.assertEqual(nested_loop(10), 275)

    def test_nested_for(self):

        @matx.script
        def nested_for() -> int:
            ss = 0
            for i in range(10):
                for j in range(10):
                    ss += 1
            return ss

        self.assertEqual(nested_for(), 100)

    def test_unpack_auto_for(self):
        @matx.script
        def auto_for(d: matx.Dict) -> int:
            ret = 0
            for k, v in d.items():
                if k % 2 == 0:
                    ret += v
            return ret

        d = {i: i * 2 for i in range(5)}
        self.assertEqual(auto_for(d), 12)

    def test_range_step(self):

        def for_with_neg_step1() -> int:
            res = 0
            for i in range(10, -1, -1):
                res += i
            return res
        py_res = for_with_neg_step1()
        tx_res = matx.script(for_with_neg_step1)()
        self.assertEqual(py_res, tx_res)

        def for_with_neg_step2() -> int:
            s = 0
            for i in range(5, 1, -2):
                s += i
            return s
        py_res = for_with_neg_step2()
        tx_res = matx.script(for_with_neg_step2)()
        self.assertEqual(py_res, tx_res)

        def for_with_neg_step3(step: int) -> int:
            s = 0
            for i in range(5, 1, step):
                s += i
            return s
        py_res = for_with_neg_step3(1)
        tx_res = matx.script(for_with_neg_step3)(1)
        self.assertEqual(py_res, tx_res)
        py_res = for_with_neg_step3(-1)
        tx_res = matx.script(for_with_neg_step3)(-1)
        self.assertEqual(py_res, tx_res)

        def for_with_pos_step() -> int:
            res = 0
            for i in range(10, 20, 1):
                res += i
            return res
        py_res = for_with_pos_step()
        tx_res = matx.script(for_with_pos_step)()
        self.assertEqual(py_res, tx_res)

    def test_max_range(self):
        @matx.script
        def end_range(l: List, end: int) -> List:
            ret = []
            for i in range(end):
                ret.append(l[i])
            return ret

        @matx.script
        def start_end_range(l: List, start: int, end: int) -> List:
            ret = []
            for i in range(start, end):
                ret.append(l[i])
            return ret

        @matx.script
        def start_end_step_range(l: List, start: int, end: int, step: int) -> List:
            ret = []
            for i in range(start, end, step):
                ret.append(l[i])
            return ret

        l = list(range(10))
        self.assertListEqual([0, 1, 2, 3, 4], list(end_range(l, 5)))
        self.assertListEqual([2, 3, 4], list(start_end_range(l, 2, 5)))
        self.assertListEqual([0, 2, 4, 6, 8], list(start_end_step_range(l, 0, 10, 2)))

    def test_step_parse(self):
        class RangeStep:
            def __init__(self, step: int) -> None:
                self.step: int = step

            def __call__(self, l: List) -> List:
                ret = []
                for i in range(0, len(l), self.step):
                    ret.append(l[i])
                return ret

        op_constructor = matx.script(RangeStep)
        op_step2 = op_constructor(2)
        self.assertListEqual([0, 2, 4, 6, 8], list(op_step2(list(range(10)))))

    def test_change_in_for(self):
        @matx.script
        def change() -> List:
            ret = []
            for i in range(5):
                i += 1
                ret.append(i)
            return ret

        ret = change()
        self.assertListEqual([1, 2, 3, 4, 5], list(ret))

    def test_advanced_if(self):
        @matx.script
        def if_list(a: List) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_set(a: Set) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_dict(a: Dict) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_tuple(a: Tuple[int, str]) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_str(a: str) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_bytes(a: bytes) -> bool:
            if a:
                return True
            return False

        @matx.script
        def if_ftlist(a: List[int]) -> bool:
            b: FTList[int] = []
            for x in a:
                b.append(x)
            if b:
                return True
            return False

        @matx.script
        def if_ftset(a: Set[int]) -> bool:
            b: FTSet[int] = set()
            for x in a:
                b.add(x)
            if b:
                return True
            return False

        @matx.script
        def if_ftdict(a: Dict[int, int]) -> bool:
            b: FTDict[int, int] = {}
            for k, v in a.items():
                b[k] = v
            if a:
                return True
            return False

        @matx.script
        def if_object(a: Any) -> bool:
            if a:
                return True
            return False

        self.assertTrue(if_list([1, 2, 3]))
        self.assertTrue(if_set({1, 2, 3}))
        self.assertTrue(if_dict({1: 2, 2: 3}))
        self.assertTrue(if_tuple((1, 2, 3)))
        self.assertTrue(if_str('123'))
        self.assertTrue(if_bytes(b'123'))
        self.assertTrue(if_ftlist([1, 2, 3]))
        self.assertTrue(if_ftset({1, 2, 3}))
        self.assertTrue(if_ftdict({1: 2, 2: 3}))

        self.assertTrue(if_object([1, 2, 3]))
        self.assertTrue(if_object({1, 2, 3}))
        self.assertTrue(if_object({1: 2, 2: 3}))
        self.assertTrue(if_object((1, 2, 3)))
        self.assertTrue(if_object('123'))
        self.assertTrue(if_object(b'123'))

        self.assertFalse(if_list([]))
        self.assertFalse(if_set(set()))
        self.assertFalse(if_dict({}))
        self.assertFalse(if_str(''))
        self.assertFalse(if_bytes(b''))
        self.assertFalse(if_ftlist([]))
        self.assertFalse(if_ftset(set()))
        self.assertFalse(if_ftdict({}))

        self.assertFalse(if_object([]))
        self.assertFalse(if_object(set()))
        self.assertFalse(if_object({}))
        self.assertFalse(if_object(''))
        self.assertFalse(if_object(b''))

    def test_temp_var(self):
        class MyData:

            def __init__(self):
                self.data: List[str] = []

            def foo(self) -> List[str]:
                self.data.append("hi")
                return self.data

            def __call__(self, ) -> int:
                s = 0
                for i in range(len(self.foo())):
                    print(i)
                    s += i
                return s

        print(MyData()())
        print(matx.script(MyData)()())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
