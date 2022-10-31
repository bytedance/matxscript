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
from typing import Any, Tuple, Callable


class MyBasicData:
    def __init__(self, a: int) -> None:
        self._a: int = a

    def get_a(self) -> int:
        return self._a

    def get_a_wrapper(self) -> int:
        a = self.get_a
        return a()


def call_any_attr(a: Any) -> int:
    func: Callable = a.get_a
    d = func()
    return d


def call_cls_attr(a: MyBasicData) -> int:
    func: Callable = a.get_a
    d = func()
    return d


def my_pipeline() -> Tuple[int, int]:
    a0 = MyBasicData(10)
    b0 = call_any_attr(a0)
    b1 = call_cls_attr(a0)
    return b0, b1


class TestClassMethodAsVar(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_class_method_as_var(self):
        py_ret = my_pipeline()
        tx_ret = matx.script(my_pipeline)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
