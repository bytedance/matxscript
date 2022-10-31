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

import os
import unittest
from abc import ABC, abstractmethod, ABCMeta

import matx


class TestInterface(unittest.TestCase):
    def test_common_usage(self):
        class MyInterface(ABC):
            # TODO: allow class with no __init__
            def __init__(self) -> None:
                pass

            @abstractmethod
            def func(self, text: str) -> int:
                pass

        class MyData(MyInterface):

            def __init__(self) -> None:
                pass

            def func(self, text: str) -> int:
                return len(text)

            def __call__(self, text: str) -> int:
                return self.func(text)

        def wrapper() -> int:
            d = MyData()
            return d('abc')

        self.assertEqual(wrapper(), matx.script(wrapper)())
        self.assertEqual(MyData()('abc'), matx.script(MyData)()('abc'))

    def test_instantiate_abc(self):
        class MyInterface(ABC):
            # TODO: allow class with no __init__
            def __init__(self) -> None:
                pass

            @abstractmethod
            def func(self, text: str) -> int:
                return 5

        class MyBadData(MyInterface):

            def __init__(self) -> None:
                pass

            def __call__(self, text: str) -> int:
                return self.func(text)

        def wrapper() -> int:
            d = MyBadData()
            return d('abc')

        with self.assertRaises(Exception):
            matx.script(wrapper)()
        with self.assertRaises(Exception):
            matx.script(MyBadData)()('abc')


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
