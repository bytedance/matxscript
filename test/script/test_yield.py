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


class TestYield(unittest.TestCase):

    def test_yield_int(self):
        @matx.script
        def yield_int(n: int) -> iter:
            for i in range(n):
                yield i

        result = [x for x in yield_int(3)]
        self.assertEqual(result, [0, 1, 2])

    def test_yield_float(self):
        @matx.script
        def yield_float(n: int) -> iter:
            for i in range(n):
                yield i + 0.1

        result = [x for x in yield_float(3)]
        expect_result = [0.1, 1.1, 2.1]
        for a, b in zip(result, expect_result):
            self.assertAlmostEqual(a, b)

    def test_complex(self):
        def test_yield_str(n: int) -> str:
            a = "hello"
            for i in range(n):
                yield a

        def test_yield_mix(n: int):
            for i in range(n):
                yield i, "hello"

        # TODO(maxiandi) : fix complex


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
