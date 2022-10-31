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
from typing import Tuple, Dict, List
import os
import json
import inspect


def factorial(n: int) -> int:
    if n <= 1:
        return n

    return n * factorial(n - 1)


def factorial_add(a: int, b: int) -> int:
    m = factorial(a)
    n = factorial(b)

    return m + n


def factorial_sub(a: int, b: int) -> int:
    m = factorial(a)
    n = factorial(b)

    return m - n


def sub(a: int, b: int) -> int:
    return a - b


factorial_add_op = matx.script(factorial_add)
factorial_sub_op = matx.script(factorial_sub)
sub_op = matx.script(sub)


def process(a: int, b: int) -> int:
    m = factorial_add_op(a, b)
    n = factorial_sub_op(a, b)

    return sub_op(m, n)


class TestCodeStat(unittest.TestCase):
    def test_code_stat(self):
        save_folder = "code_stat_tmp"
        m = matx.pipeline.Trace(process, 10, 10)
        m.save(save_folder)

        with open(os.path.join(save_folder, "code_stat_info.json")) as f:
            code_stat_info = json.load(f)
            self.assertEqual(code_stat_info["jit_op_count"], 3)
            self.assertEqual(17, code_stat_info["python_co_lines"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
