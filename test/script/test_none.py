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


class TestNone(unittest.TestCase):

    def test_none(self):
        @matx.script
        def none_as_param(a: Any) -> int:
            if a is None:
                return 0
            return 1

        @matx.script
        def none_as_ret() -> None:
            return None

        @matx.script
        def none_should_be_object() -> Any:
            a = None
            a = 12
            return a

        self.assertEqual(none_as_param(None), 0)
        self.assertEqual(none_as_param(123), 1)
        self.assertIs(none_as_ret(), None)
        self.assertEqual(none_should_be_object(), 12)

    def test_default_param(self):
        @matx.script
        def generic_no_enough_params(s: Any) -> List:
            return s.split()

        @matx.script
        def generic_none_param_eval(s: Any) -> List:
            return s.split(None)

        @matx.script
        def explicit_no_enough_params(s: str) -> List:
            return s.split()

        @matx.script
        def explicit_none_param_eval(s: str) -> List:
            return s.split(None)

        self.assertEqual(generic_no_enough_params('a b c'), ['a', 'b', 'c'])
        self.assertEqual(generic_none_param_eval('a b c'), ['a', 'b', 'c'])
        self.assertEqual(explicit_no_enough_params('a b c'), ['a', 'b', 'c'])
        self.assertEqual(explicit_none_param_eval('a b c'), ['a', 'b', 'c'])

    def test_none_as_class_init_arg(self):
        class MyClass:
            def __init__(self, s: Any) -> None:
                self.s: Any = s

        matx.script(MyClass)(None)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
