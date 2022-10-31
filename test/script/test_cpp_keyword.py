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

from typing import Tuple


class TestCppKeyword(unittest.TestCase):

    def test_normalize_func(self):

        @matx.script
        def cpp_kw_as_var() -> None:
            int = '3'

        @matx.script
        def cpp_kw_as_arg() -> int:
            int = '3'
            return int(int)

        @matx.script
        def cpp_kw_as_param(int: int) -> int:
            return int + 3

        self.assertEqual(cpp_kw_as_var(), None)
        self.assertEqual(cpp_kw_as_arg(), 3)
        self.assertEqual(cpp_kw_as_param(5), 8)

    def test_normailze_class(self):
        class Op:
            __slots__: Tuple[int] = ['protected']

            def __init__(self, v: int) -> None:
                self.protected = v

            def forward(self) -> int:
                return self.protected

        matx.script(Op)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
