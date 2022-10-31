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
from typing import Dict, List


class TestUserClass(unittest.TestCase):

    def test_empty_class(self):
        class EmptyClass:
            def __init__(self) -> None:
                pass

            def __call__(self) -> str:
                return "hello world!"

        empty_op = matx.script(EmptyClass)()
        assert empty_op() == "hello world!"

    def test_normal_class(self):
        class ClassWithAttr:
            def __init__(self) -> None:
                self.word2id: Dict[str, int] = {'a': 1, 'b': 2}
                self.offset: int = 10
                if True:
                    self.k: int = 5

            def __call__(self) -> int:
                return self.word2id['a'] + self.word2id['b'] + self.offset + self.k

        scripted_op = matx.script(ClassWithAttr)()
        assert scripted_op() == 18

    def test_class_subn_ann(self):
        class ClassWithSubAnn:
            def __init__(self) -> None:
                self.y: List[int] = [1, 2, 3]
                self.y[0]: int = 10
                a: int = self.y[0]

            def __call_(self) -> int:
                return self.y[0]

        matx.script(ClassWithSubAnn)()

    def test_class_call_attr(self):
        class MySimpleClass:
            def __init__(self) -> None:
                pass

            def func1(self) -> float:
                return 1.0

            def func2(self) -> float:
                return 2.0

        py_obj = MySimpleClass()
        compiled_obj = matx.script(MySimpleClass)()
        self.assertAlmostEqual(py_obj.func1(), compiled_obj.func1())
        self.assertAlmostEqual(py_obj.func2(), compiled_obj.func2())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
