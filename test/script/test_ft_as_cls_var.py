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
from matx import FTList
from typing import Any


class ClassWithFullTypedVar:

    def __init__(self) -> None:
        self.b: FTList[int] = [1, 2, 3]

    def foo_sum(self) -> int:
        s = 0
        for i in self.b:
            s += i
        return s


class TestClassWithFullTypedVar(unittest.TestCase):

    def test_normal_class_full_typed_var(self):
        result = matx.script(ClassWithFullTypedVar)().foo_sum()
        self.assertEqual(result, ClassWithFullTypedVar().foo_sum())

        def my_explicit_access_var() -> int:
            a = ClassWithFullTypedVar()
            s = 0
            for i in a.b:
                s += i
            return s

        result = matx.script(my_explicit_access_var)()
        self.assertEqual(result, my_explicit_access_var())

    def test_class_full_typed_var_generic_access(self):
        def my_generic_access_var() -> int:
            a = ClassWithFullTypedVar()
            obj: Any = a
            s = 0
            for i in obj.b:
                s += i
            return s

        def my_generic_set_var() -> None:
            a = ClassWithFullTypedVar()
            obj: Any = a
            b: FTList[int] = []
            obj.b = b

        def my_explicit_set_var_as_other_var() -> None:
            a = ClassWithFullTypedVar()
            xx = []
            a.b = xx

        def my_explicit_set_var_constant() -> None:
            a = ClassWithFullTypedVar()
            a.b = []

        matx.script(my_generic_access_var)()
        matx.script(my_generic_set_var)()

        with self.assertRaises(Exception):
            matx.script(my_explicit_set_var_as_other_var)

        matx.script(my_explicit_set_var_constant)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
