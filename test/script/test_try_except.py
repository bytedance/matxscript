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


class TestTryExcept(unittest.TestCase):

    def test_try_no_throw(self):
        def try_no_throw() -> str:
            try:
                a = 0
            except:
                return "error"
            return "ok"

        py_ret = try_no_throw()
        print(py_ret)
        tx_ret = matx.script(try_no_throw)()
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_try_and_throw(self):
        def try_and_throw() -> str:
            try:
                a = int("xxx")
            except:
                return "error"
            return "ok"

        py_ret = try_and_throw()
        print(py_ret)
        tx_ret = matx.script(try_and_throw)()
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_try_reporter(self):
        def try_except_specific1() -> str:
            try:
                a = int("xxx")
            except BaseException:
                return "error"
            return "ok"

        with self.assertRaises(BaseException):
            matx.script(try_except_specific1)()

        def try_except_specific2() -> str:
            try:
                a = int("xxx")
            except BaseException as e:
                return "error"
            return "ok"

        with self.assertRaises(BaseException):
            matx.script(try_except_specific2)()

        def try_except_finally() -> str:
            try:
                a = int("xxx")
            except BaseException as e:
                return "error"
            finally:
                b = 0
            return "ok"

        with self.assertRaises(BaseException):
            matx.script(try_except_finally)()

    def test_throw_exception(self):
        def throw_exception() -> None:
            raise Exception("error")

        tx_func = matx.script(throw_exception)
        with self.assertRaises(Exception):
            tx_func()

    def test_not_supported_throw_exception(self):
        def not_supported_throw_exception() -> None:
            raise ImportError("error")

        with self.assertRaises(BaseException):
            matx.script(not_supported_throw_exception)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
