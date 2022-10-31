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
from typing import Any, List


@matx.script
def my_func(x: List[str]) -> List[str]:
    return [v.lower() for v in x]


def my_py_func(x: List[str]) -> List[str]:
    return [v.lower() for v in x]


class MyFunctor:

    def __init__(self):
        self.r: int = 1

    def map_func(self, v: List[int]) -> List[int]:
        return [self.r * x for x in v]

    def __call__(self, v: List[int]) -> List[int]:
        return matx.apply_async(self.map_func, v).get()


class TestAutoApplyAsync(unittest.TestCase):

    def test_script_async(self):
        def my_entry() -> Any:
            a = ["Hello", "World"] * 10
            b = matx.apply_async(my_func, a)
            return b.get()

        py_ret = my_entry()
        tx_ret = matx.script(my_entry)()
        self.assertEqual(py_ret, tx_ret)

    def test_py_apply_async(self):
        def my_entry() -> Any:
            a = ["Hello", "World"] * 10
            b = matx.apply_async(my_py_func, a)
            return b.get()

        py_ret = my_entry()
        tx_ret = matx.script(my_entry)()
        self.assertEqual(py_ret, tx_ret)

    def test_functor_apply_async(self):
        a = list(range(10))
        py_ret = MyFunctor()(a)
        tx_ret = matx.script(MyFunctor)()(a)
        self.assertEqual(py_ret, tx_ret)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
