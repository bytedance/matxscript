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
from typing import List, Any, Dict


class TestZip(unittest.TestCase):

    def test_zip_list(self):
        def zip_two_list(a: List[int], b: List[int]) -> Dict[int, int]:
            return {x: y for x, y in zip(a, b)}

        input_a = [1, 2, 3] * 10
        input_b = [4, 5, 6] * 10
        py_ret = zip_two_list(input_a, input_b)
        print(py_ret)
        tx_ret = matx.script(zip_two_list)(input_a, input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_zip_any(self):
        def zip_two_any(a: Any, b: Any) -> Any:
            return {x: y for x, y in zip(a, b)}

        input_a = [1, 2, 3] * 10
        input_b = [4, 5, 6] * 10
        py_ret = zip_two_any(input_a, input_b)
        print(py_ret)
        tx_ret = matx.script(zip_two_any)(input_a, input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_zip_str(self):
        def zip_two_str(a: str, b: str) -> Dict[str, str]:
            return {x: y for x, y in zip(a, b)}

        input_a = "abc"
        input_b = "def"
        py_ret = zip_two_str(input_a, input_b)
        print(py_ret)
        tx_ret = matx.script(zip_two_str)(input_a, input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_zip_return_one(self):
        def zip_two_list_return_tup(a: List[int], b: Any) -> Any:
            return [x for x in zip(a, b)]

        def zip_two_str_return_tup(a: str, b: str) -> Any:
            return [x for x in zip(a, b)]

        input_a = [1, 2, 3] * 10
        input_b = [4, 5, 6] * 10
        py_ret = zip_two_list_return_tup(input_a, input_b)
        print(py_ret)
        tx_ret = matx.script(zip_two_list_return_tup)(input_a, input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

        input_a = "abc"
        input_b = "def"
        py_ret = zip_two_str_return_tup(input_a, input_b)
        print(py_ret)
        tx_ret = matx.script(zip_two_str_return_tup)(input_a, input_b)
        print(tx_ret)

        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
