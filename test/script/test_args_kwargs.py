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


class TestArgsKwargs(unittest.TestCase):

    def test_args(self):
        def py_with_args(*args) -> None:
            return None

        with self.assertRaises(Exception):
            matx.script(py_with_args)

    def test_kwargs(self):
        def py_with_kwargs(**kwargs) -> None:
            return None

        with self.assertRaises(Exception):
            matx.script(py_with_kwargs)

    def test_args_kwargs(self):
        def py_with_args_kwargs(*args, **kwargs) -> None:
            return None

        with self.assertRaises(Exception):
            matx.script(py_with_args_kwargs)

    def test_kwonly(self):
        def py_with_kwonly(*, a: int = 1) -> None:
            return None

        with self.assertRaises(Exception):
            matx.script(py_with_kwonly)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
