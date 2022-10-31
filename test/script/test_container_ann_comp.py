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
from typing import List, Dict, Set
from matx import FTList, FTSet, FTDict


class TestAnnotationContainerComprehension(unittest.TestCase):

    def setUp(self) -> None:
        self.func_holder = []  # keep func alive before data is del

    def test_list_comp(self):
        def convert_input(a: List[List[int]]) -> FTList[FTList[int]]:
            a_ft: FTList[FTList[int]] = [
                [i for i in item] for item in a
            ]
            return a_ft

        input_a = [[1, 2, 3]]
        py_ret = convert_input(input_a)
        print(py_ret)
        tx_func = matx.script(convert_input)
        self.func_holder.append(tx_func)
        tx_ret = tx_func(input_a)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)
        del tx_ret
        del tx_func

    def test_set_comp(self):
        def convert_input(a: List[List[int]]) -> FTList[FTSet[int]]:
            a_ft: FTList[FTSet[int]] = [
                {i for i in item} for item in a
            ]
            return a_ft

        input_a = [[1, 2, 3]]
        py_ret = convert_input(input_a)
        print(py_ret)
        tx_func = matx.script(convert_input)
        self.func_holder.append(tx_func)
        tx_ret = tx_func(input_a)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)
        del tx_ret
        del tx_func

    def test_dict_comp(self):
        def convert_input(a: List[List[int]]) -> FTDict[int, FTList[FTList[int]]]:
            a_ft: FTDict[int, FTList[FTList[int]]] = {
                k: [
                    [i for i in item] for item in a
                ] for k in range(10)
            }
            return a_ft

        input_a = [[1, 2, 3]]
        py_ret = convert_input(input_a)
        print(py_ret)
        tx_func = matx.script(convert_input)
        self.func_holder.append(tx_func)
        tx_ret = tx_func(input_a)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)
        del tx_ret
        del tx_func


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
