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
from typing import Any, Dict


class ToyPt:

    def __init__(self) -> None:
        self.i: int = 1

    def __call__(self, target_key: bytes) -> int:
        print(self)
        r = self.i
        print("toypt forward:", r, target_key)
        self.i += 1
        return r


class FusedOp:
    def __init__(self, x: ToyPt) -> None:
        x(b"haha")
        self.toy_pt_op: ToyPt = x

    def __call__(self, target_key: bytes) -> int:
        print(self.toy_pt_op)
        return self.toy_pt_op(target_key)


class TestNestedClass(unittest.TestCase):

    def test_nested_class(self):
        toy_pt_op = matx.script(ToyPt)()
        toy_pt_op(b"cc")
        fused_op_creator = matx.script(FusedOp)
        fused_op = fused_op_creator(toy_pt_op)
        fused_op(b"bb")


class MyClass1:
    def __init__(self) -> None:
        self.init: int = 1

    def func(self, i: int) -> int:
        return i + self.init


class MyClass2:
    def __init__(self, d: Dict) -> None:
        self.d: Dict = d
        self.obj: Any = d["c1"]

    def __call__(self, i: int) -> int:
        return self.obj.func(i)


class TestJitObject(unittest.TestCase):

    def test_jit_object_in_container(self):
        d = matx.Dict({'c1': matx.script(MyClass1)()})
        r = matx.script(MyClass2)(d)(10)
        r = matx.script(MyClass2)(d)(10)
        self.assertEqual(r, 11)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
