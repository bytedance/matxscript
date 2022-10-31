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


class A:
    def __init__(self) -> None:
        pass


class B:
    def __init__(self, param: Any) -> None:
        self.sub_class: A = param


class MyFunctor:
    def __init__(self, a: int) -> None:
        self._a: int = a

    def get_a(self) -> int:
        return self._a

    def __call__(self) -> int:
        return self._a


class CallAnyAttr:
    def __init__(self, a: Any) -> None:
        self.b: int = a.get_a()


class TestAnyToUserObject(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_to_object(self):
        a = matx.script(A)()
        matx.script(B)(a)
        a0 = matx.script(MyFunctor)(10)
        b0 = matx.script(CallAnyAttr)(a0)
        print(b0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
