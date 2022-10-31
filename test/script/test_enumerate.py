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
from typing import List, Tuple, Any


class TestEnumerate(unittest.TestCase):

    def test_enumerate_list(self):
        @matx.script
        def enumerate_list(cons: List) -> Tuple[List, List]:
            ret1 = []
            ret2 = []
            for i, val in enumerate(cons):
                ret1.append(i)
                ret2.append(val)
            return ret1, ret2

        test_cons = [1, 2, 3]
        r1, r2 = enumerate_list(test_cons)
        self.assertEqual(r1, [0, 1, 2])
        self.assertEqual(r2, test_cons)

    def test_enumerate_any(self):
        @matx.script
        def enumerate_object(cons: Any) -> Tuple[List, List]:
            ret1 = []
            ret2 = []
            for i, val in enumerate(cons):
                ret1.append(i)
                ret2.append(val)
            return ret1, ret2

        test_cons = [1, 2, 3]
        r1, r2 = enumerate_object(test_cons)
        self.assertEqual(r1, [0, 1, 2])
        self.assertEqual(r2, test_cons)

    def test_enumerate_str(self):
        @matx.script
        def enumerate_str(cons: str) -> Tuple[List, List]:
            ret1 = []
            ret2 = []
            for i, val in enumerate(cons):
                ret1.append(i)
                ret2.append(val)
            return ret1, ret2

        test_cons = "abc"
        r1, r2 = enumerate_str(test_cons)
        self.assertEqual(r1, [0, 1, 2])
        self.assertEqual(r2, ['a', 'b', 'c'])

    def test_enumerate_return_one(self):
        @matx.script
        def enumerate_list_return_tup(a: List[int]) -> Any:
            return [x for x in enumerate(a)]

        test_cons = [1, 2, 3]
        r = enumerate_list_return_tup(test_cons)
        self.assertEqual(r, [(0, 1), (1, 2), (2, 3)])

        @matx.script
        def enumerate_str_return_tup(a: str) -> Any:
            return [x for x in enumerate(a)]

        test_cons = "abc"
        r = enumerate_str_return_tup(test_cons)
        self.assertEqual(r, [(0, 'a'), (1, 'b'), (2, 'c')])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
