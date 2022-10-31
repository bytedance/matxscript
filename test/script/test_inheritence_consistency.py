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
from typing import Tuple, Any
import unittest
import matx


class TestInheritenceConsistencyCheck(unittest.TestCase):
    def test_var(self):
        class A:
            def __init__(self) -> None:
                pass

            def func(self) -> int:
                return 1

        class B(A):
            def __init__(self) -> None:
                super().__init__()
                self.var: str = 'b'

        class C(B):
            def __init__(self) -> None:
                super().__init__()
                self.var: int = 1

        with self.assertRaises(Exception):
            matx.script(C)

    def test_func(self):
        class A:
            def __init__(self) -> None:
                pass

            def func(self) -> int:
                return 1

        class B(A):
            def __init__(self) -> None:
                super().__init__()
                self.var: int = 2

            def func(self) -> str:
                return 'b'

        class C(B):
            def __init__(self) -> None:
                super().__init__()
                self.var: int = 1

        with self.assertRaises(Exception):
            matx.script(C)

    def test_func_defaults(self):
        class A:
            def __init__(self) -> None:
                pass

            def func(self, val: int = 1) -> int:
                return val

        class B(A):
            def __init__(self) -> None:
                pass

            def func(self, val: int = 2) -> int:
                return val

        with self.assertRaises(Exception):
            matx.script(B)

    def test_func_conflict_with_var(self):
        class A:
            def __init__(self) -> None:
                self.val: int = 2

            def func(self) -> int:
                return 1

        class B(A):
            def __init__(self) -> None:
                self.func: int = 3

        class C(A):
            def __init__(self) -> None:
                pass

            def val(self) -> int:
                return 2

        with self.assertRaises(Exception):
            matx.script(B)
        with self.assertRaises(Exception):
            matx.script(C)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
