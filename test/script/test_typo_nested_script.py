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


class TestTypoNestedScript(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_nested_script(self):
        class MyData:
            def __init__(self):
                pass

        class MyFunctor:
            def __init__(self):
                pass

            def __call__(self) -> Any:
                return None

        def my_fn(s: str) -> str:
            return "[CLS] " + s

        matx.script(matx.script(my_fn))
        matx.script(matx.script(MyData))
        matx.script(matx.script(MyFunctor))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
