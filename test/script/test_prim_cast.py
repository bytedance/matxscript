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


class TestPrimTypeConversion(unittest.TestCase):

    def test_convert_to_int(self):
        @matx.script
        def int_to_int(a: int) -> int:
            return int(a)

        @matx.script
        def float_to_int(a: float) -> int:
            return int(a)

        @matx.script
        def bool_to_int(a: bool) -> int:
            return int(a)

        @matx.script
        def str_to_int(a: str) -> int:
            return int(a)

        @matx.script
        def str_to_int_with_base(a: str, base: int) -> int:
            return int(a, base)

        @matx.script
        def bytes_to_int(a: bytes) -> int:
            return int(a)

        @matx.script
        def default_int() -> int:
            return int()

        self.assertEqual(int_to_int(2), 2)
        self.assertEqual(float_to_int(2.0), 2)
        self.assertEqual(bool_to_int(True), 1)
        self.assertEqual(bool_to_int(False), 0)

        self.assertEqual(str_to_int('-0'), 0)
        self.assertEqual(str_to_int('  +10 '), 10)
        self.assertEqual(str_to_int('-010'), -10)
        self.assertEqual(str_to_int(' -0010'), -10)
        self.assertEqual(str_to_int_with_base('0o10', 8), 8)
        self.assertEqual(str_to_int_with_base(' +0010 ', 10), 10)
        self.assertEqual(str_to_int_with_base('+0x10', 16), 16)
        self.assertEqual(str_to_int_with_base('-0O10', 8), -8)
        self.assertEqual(str_to_int_with_base('-010', 10), -10)
        self.assertEqual(str_to_int_with_base('-0X10', 16), -16)
        self.assertEqual(str_to_int_with_base('+11', 0), 11)
        self.assertEqual(str_to_int_with_base('0b11', 0), 3)
        self.assertEqual(str_to_int_with_base('0o11', 0), 9)
        self.assertEqual(str_to_int_with_base('0O11', 0), 9)
        self.assertEqual(str_to_int_with_base('0x11', 0), 17)
        self.assertEqual(str_to_int_with_base('0X11', 0), 17)
        self.assertEqual(str_to_int_with_base('11', 9), 10)
        self.assertEqual(str_to_int_with_base('0011', 2), 3)
        self.assertEqual(default_int(), 0)

        self.assertEqual(bytes_to_int(b'+10'), 10)
        self.assertEqual(bytes_to_int(b'-010'), -10)

        with self.assertRaises(Exception) as context:
            str_to_int('')
        with self.assertRaises(Exception) as context:
            str_to_int('4.')
        with self.assertRaises(Exception) as context:
            str_to_int('.53')
        with self.assertRaises(Exception) as context:
            str_to_int('2.56')
        with self.assertRaises(Exception) as context:
            str_to_int_with_base('0X10', 8)
        with self.assertRaises(Exception) as context:
            str_to_int_with_base('x10', 16)
        with self.assertRaises(Exception) as context:
            str_to_int_with_base('x10', 10)
        with self.assertRaises(Exception) as context:
            str_to_int_with_base('x10', 0)
        with self.assertRaises(Exception) as context:
            str_to_int_with_base(' 0011 ', 0)

        with self.assertRaises(Exception) as context:
            bytes_to_int(b'')
        with self.assertRaises(Exception) as context:
            bytes_to_int(b'4.')
        with self.assertRaises(Exception) as context:
            bytes_to_int(b'.53')
        with self.assertRaises(Exception) as context:
            bytes_to_int(b'2.56')

    def test_convert_to_float(self):
        @matx.script
        def int_to_float(a: int) -> float:
            return float(a)

        @matx.script
        def float_to_float(a: float) -> float:
            return float(a)

        @matx.script
        def bool_to_float(a: bool) -> float:
            return float(a)

        @matx.script
        def str_to_float(a: str) -> float:
            return float(a)

        @matx.script
        def bytes_to_float(a: bytes) -> float:
            return float(a)

        self.assertAlmostEqual(int_to_float(2), 2.0)
        self.assertAlmostEqual(float_to_float(2.0), 2.0)
        self.assertAlmostEqual(bool_to_float(True), 1.0)
        self.assertAlmostEqual(bool_to_float(False), 0.0)

        self.assertAlmostEqual(str_to_float('2'), 2.0)
        self.assertAlmostEqual(str_to_float('.2'), 0.2)
        self.assertAlmostEqual(str_to_float('2.5'), 2.5)
        self.assertAlmostEqual(str_to_float('02.5'), 2.5)
        self.assertAlmostEqual(str_to_float('-2'), -2.0)
        self.assertAlmostEqual(str_to_float('-2.5'), -2.5)
        self.assertAlmostEqual(str_to_float('-02.5'), -2.5)

        self.assertAlmostEqual(bytes_to_float(b'2'), 2.0)
        self.assertAlmostEqual(bytes_to_float(b'.2'), 0.2)
        self.assertAlmostEqual(bytes_to_float(b'2.5'), 2.5)
        self.assertAlmostEqual(bytes_to_float(b'02.5'), 2.5)
        self.assertAlmostEqual(bytes_to_float(b'-2'), -2.0)
        self.assertAlmostEqual(bytes_to_float(b'-2.5'), -2.5)
        self.assertAlmostEqual(bytes_to_float(b'-02.5'), -2.5)

        with self.assertRaises(Exception) as context:
            str_to_float('')
        with self.assertRaises(Exception) as context:
            bytes_to_int('')


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
