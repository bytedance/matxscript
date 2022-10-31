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
from typing import Any, Tuple
from matx.pypi import farmhash as farmhash


class TestPyPiFarmhash(unittest.TestCase):

    def test_hash32(self):
        def my_hash32() -> Any:
            s = "hello"
            k = farmhash.hash32(s) % 1024576
            return k

        py_ret = my_hash32()
        tx_ret = matx.script(my_hash32)()
        self.assertEqual(py_ret, tx_ret)

    def test_hash64(self):
        def my_hash64() -> Any:
            s = "hello"
            k = farmhash.hash64_mod(s, 1024576)
            return k

        py_ret = my_hash64()
        self.assertTrue(py_ret >= 0)
        tx_ret = matx.script(my_hash64)()
        self.assertEqual(py_ret, tx_ret)

    def test_fingerprint32(self):
        def my_fingerprint32() -> Any:
            s = "hello"
            k = farmhash.fingerprint32(s) % 1024576
            return k

        py_ret = my_fingerprint32()
        tx_ret = matx.script(my_fingerprint32)()
        self.assertEqual(py_ret, tx_ret)

    def test_fingerprint64(self):
        def my_fingerprint64() -> Any:
            s = "hello"
            k = farmhash.fingerprint64_mod(s, 1024576)
            return k

        py_ret = my_fingerprint64()
        self.assertTrue(py_ret >= 0)
        tx_ret = matx.script(my_fingerprint64)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
