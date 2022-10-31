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
import os
from matx import toolchain
from typing import Any


def find_compile_log(output):
    for line in output[::-1]:
        if not ('matx compile function/class' in line or 'skip compiling' in line):
            continue
        return line
    return ''


class TestCacheSo(unittest.TestCase):

    def test_cache_hit(self):

        with self.assertLogs(level='INFO') as log:
            def func(a: int, b: int) -> int:
                return a + b

            toolchain.USE_SO_CACHE = False
            c = matx.script(func)(1, 2)
            self.assertIn('matx compile function/class', find_compile_log(log.output))
            self.assertEqual(c, 3)
            log.output.clear()

            toolchain.USE_SO_CACHE = True
            d = matx.script(func)(1, 2)
            self.assertIn('info matched, skip compiling', find_compile_log(log.output))
            self.assertEqual(d, 3)
            log.output.clear()

            func.is_build = False
            d = matx.script(func)(1, 2)
            self.assertIn('info matched, skip compiling', find_compile_log(log.output))
            self.assertEqual(d, 3)
            log.output.clear()

    def test_cache_expire(self):
        with self.assertLogs(level='INFO') as log:
            rm_files = [
                matx.toolchain.LIB_PATH +
                '/libfunc_plugin.so',
                matx.toolchain.LIB_PATH +
                '/libfunc1_plugin.so',
                matx.toolchain.LIB_PATH +
                '/libfunc2_plugin.so']
            for f in rm_files:
                if os.path.exists(f):
                    os.remove(f)
            toolchain.USE_SO_CACHE = False

            def func(a: int, b: int) -> int:
                return a + b

            scripted_fun = matx.script(func)
            c = scripted_fun(1, 2)
            self.assertIn('matx compile function/class', find_compile_log(log.output))
            self.assertEqual(c, 3)

            scripted_fun.is_build = False

            toolchain.USE_SO_CACHE = True

            def func(a: int, b: int) -> int:
                return a + b * 2

            scripted_fun = matx.script(func)
            d = scripted_fun(1, 2)
            self.assertIn('matx compile function/class', find_compile_log(log.output))
            self.assertEqual(d, 5)

    # issue: #283
    def test_unpack(self):
        with self.assertLogs(level='INFO') as log:
            def func(x: int) -> Any:
                return 1, 2 * x

            def unpack_func() -> int:
                a, b = func(1)
                c, d = func(2)
                return a + b + c + d

            toolchain.USE_SO_CACHE = False
            c = matx.script(unpack_func)()
            self.assertIn('matx compile function/class', find_compile_log(log.output))
            self.assertEqual(c, 8)
            log.output.clear()

            toolchain.USE_SO_CACHE = True
            d = matx.script(unpack_func)()
            self.assertIn('info matched, skip compiling', find_compile_log(log.output))
            self.assertEqual(d, 8)
            log.output.clear()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    unittest.main()
