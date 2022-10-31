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
from typing import Any, List, Dict, Set


class MyFoo:

    def __init__(self) -> None:
        self.i: int = 0
        self.j: float = 0.1


class TestTypeLift(unittest.TestCase):

    def test_assign_float_to_int(self):
        def assign_float_to_int() -> Any:
            a = 0
            a += 0.1
            return a

        # generic var can change type at the same level
        matx.script(assign_float_to_int)

        def assign_float_to_int(flag: bool) -> Any:
            a = 0
            if flag:
                a += 0.1
            return a

        with self.assertRaises(Exception):
            matx.script(assign_float_to_int)

        def assign_attr_float_to_int() -> Any:
            foo = MyFoo()
            foo.i += 0.1  # should report error
            return foo.i

        with self.assertRaises(Exception):
            matx.script(assign_attr_float_to_int)

        def assign_list_getitem_float_to_int() -> Any:
            foo: List[int] = [0]
            foo[0] += 0.1  # should report error
            return foo[0]

        with self.assertRaises(Exception):
            matx.script(assign_list_getitem_float_to_int)

        def assign_dict_getitem_float_to_int() -> Any:
            foo: Dict[int, int] = {0: 0}
            foo[0] += 0.1  # should report error
            return foo[0]

        with self.assertRaises(Exception):
            matx.script(assign_dict_getitem_float_to_int)

        def set_add_float_to_int() -> None:
            foo: Set[int] = {0}
            foo.add(0.1)  # should report error

        with self.assertRaises(Exception):
            matx.script(set_add_float_to_int)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
