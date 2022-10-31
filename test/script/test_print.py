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

import io
import contextlib


class TestBuiltinPrint(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_constructor(self):
        @matx.script
        def test_print() -> None:
            test_data = matx.List(["hello", 1, 0.3, matx.List(["hi"])])
            print(test_data)
            print(test_data, test_data)
            print(test_data, test_data, sep=" @@ ", end="\n")

        test_print()

    def test_float_print(self):
        @matx.script
        def test_fp() -> None:
            print(0., 1.)
            test_data = [0, 1., 0.]
            print(test_data)

        test_fp()

    def test_behavior_in_python(self) -> None:

        from test_print_script import print_func

        builtin_output = io.StringIO()
        with contextlib.redirect_stdout(builtin_output):
            print_func()

        import subprocess

        # We can't capture the output from matx with contextlib, so we use Popen
        proc = subprocess.Popen(["python3", "test_print_script.py"], stdout=subprocess.PIPE)
        out = proc.communicate()[0].decode()

        print(out)
        self.assertEqual(
            builtin_output.getvalue(),
            "abc||b'def'||1||1.2||['abc', b'def', 1, 1.2]||{1: 'abc', b'def': 1.2}||abc\n")
        self.assertEqual(
            out, "abc||b'def'||1||1.2||['abc', b'def', 1, 1.2]||{b'def': 1.2, 1: 'abc'}||abc\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
