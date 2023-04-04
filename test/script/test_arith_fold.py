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


class TestMatxArithConstFold(unittest.TestCase):

    def test_floor_mod(self):
        def floor_mod_int_int() -> Any:
            return 20 % 12, 20 % -12, -20 % 12, -20 % 1, 0 % 12

        def floor_mod_int_float() -> Any:
            return 20 % 12.1, 20 % -12.1, -20 % 12.1, -20 % 1.0, 0 % 12.1

        def floor_mod_float_int() -> Any:
            return 20.1 % 12, 20.1 % -12, -20.1 % 12, -20.1 % 1, 0.0 % 12

        def floor_mod_float_float() -> Any:
            return 20.1 % 12.1, 20.1 % -12.1, -20.1 % 12.1, -20.0 % 1.0, 0.0 % 12.1

        test_funcs = [
            floor_mod_int_int,
            floor_mod_int_float,
            floor_mod_float_int,
            floor_mod_float_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            tx_func = matx.script(f)
            self.assertEqual(f(), tx_func())

        def floor_mod_int_and_int_var(x: int) -> Any:
            return 20 % x, 20 % -x, -20 % x, 0 % x

        def floor_mod_int_and_float_var(x: float) -> Any:
            return 20 % x, 20 % -x, -20 % x, 0 % x

        def floor_mod_float_and_int_var(x: int) -> Any:
            return 20.1 % x, 20.1 % -x, -20.1 % x, 0.0 % x

        def floor_mod_float_and_float_var(x: float) -> Any:
            return 20.1 % x, 20.1 % -x, -20.1 % x, 0.0 % x

        def floor_mod_int_var_and_int(x: int) -> Any:
            return x % 20, -x % 20, x % -20, x % 1

        def floor_mod_int_var_and_float(x: float) -> Any:
            return x % 20.1, -x % 20.1, x % -20.1, x % 1.0

        def floor_mod_float_var_and_int(x: int) -> Any:
            return x % 20, -x % 20, x % -20, x % 1

        def floor_mod_float_var_and_float(x: float) -> Any:
            return x % 20.1, -x % 20.1, x % -20.1, x % 1.0

        test_funcs = [
            floor_mod_int_and_int_var,
            floor_mod_int_and_float_var,
            floor_mod_float_and_int_var,
            floor_mod_float_and_float_var,
            floor_mod_int_var_and_int,
            floor_mod_int_var_and_float,
            floor_mod_float_var_and_int,
            floor_mod_float_var_and_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            ann = f.__annotations__.get("x")
            input_x = ann(10.1)
            tx_func = matx.script(f)
            self.assertEqual(f(input_x), tx_func(input_x))

    def test_floor_div(self):
        def floor_div_int_int() -> Any:
            return 20 // 12, 20 // -12, -20 // 12, -20 // 12, -20 // 1, 0 // 12

        def floor_div_int_float() -> Any:
            return 20 // 12.1, 20 // -12.1, -20 // 12.1, -20 // 1.0, 0 // 12.1

        def floor_div_float_int() -> Any:
            return 20.1 // 12, 20.1 // -12, -20.1 // 12, -20.1 // 1, 0.0 // 12

        def floor_div_float_float() -> Any:
            return 20.1 // 12.1, 20.1 // -12.1, -20.1 // 12.1, -20.1 // 1.0, 0.0 // 12.1

        test_funcs = [
            floor_div_int_int,
            floor_div_int_float,
            floor_div_float_int,
            floor_div_float_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            tx_func = matx.script(f)
            self.assertEqual(f(), tx_func())

        def floor_div_int_and_int_var(x: int) -> Any:
            return 20 // x, 20 // -x, -20 // x, 0 // x

        def floor_div_int_and_float_var(x: float) -> Any:
            return 20 // x, 20 // -x, -20 // x, 0 // x

        def floor_div_float_and_int_var(x: int) -> Any:
            return 20.1 // x, 20.1 // -x, -20.1 // x, 0.0 // x

        def floor_div_float_and_float_var(x: float) -> Any:
            return 20.1 // x, 20.1 // -x, -20.1 // x, 0.0 // x

        def floor_div_int_var_and_int(x: int) -> Any:
            return x // 20, -x // 20, x // -20, x // 1

        def floor_div_int_var_and_float(x: float) -> Any:
            return x // 20.1, -x // 20.1, x // -20.1, x // 1.0

        def floor_div_float_var_and_int(x: int) -> Any:
            return x // 20, -x // 20, x // -20, x // 1

        def floor_div_float_var_and_float(x: float) -> Any:
            return x // 20.1, -x // 20.1, x // -20.1, x // 1.0

        test_funcs = [
            floor_div_int_and_int_var,
            floor_div_int_and_float_var,
            floor_div_float_and_int_var,
            floor_div_float_and_float_var,
            floor_div_int_var_and_int,
            floor_div_int_var_and_float,
            floor_div_float_var_and_int,
            floor_div_float_var_and_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            ann = f.__annotations__.get("x")
            input_x = ann(10.1)
            tx_func = matx.script(f)
            if f(input_x) != tx_func(input_x):
                print(1)
            self.assertEqual(f(input_x), tx_func(input_x))

    def test_div(self):
        def div_int_int() -> Any:
            return 20 / 12, 20 / -12, -20 / 12, -20 / 12, -20 / 1, 0 / 12

        def div_int_float() -> Any:
            return 20 / 12.1, 20 / -12.1, -20 / 12.1, -20 / 1.0, 0 / 12.1

        def div_float_int() -> Any:
            return 20.1 / 12, 20.1 / -12, -20.1 / 12, -20.1 / 1, 0.0 / 12

        def div_float_float() -> Any:
            return 20.1 / 12.1, 20.1 / -12.1, -20.1 / 12.1, -20.1 / 1.0, 0.0 / 12.1

        test_funcs = [
            div_int_int,
            div_int_float,
            div_float_int,
            div_float_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            tx_func = matx.script(f)
            self.assertEqual(f(), tx_func())

        def div_int_and_int_var(x: int) -> Any:
            return 20 / x, 20 / -x, -20 / x, 0 / x

        def div_int_and_float_var(x: float) -> Any:
            return 20 / x, 20 / -x, -20 / x, 0 / x

        def div_float_and_int_var(x: int) -> Any:
            return 20.1 / x, 20.1 / -x, -20.1 / x, 0.0 / x

        def div_float_and_float_var(x: float) -> Any:
            return 20.1 / x, 20.1 / -x, -20.1 / x, 0.0 / x

        def div_int_var_and_int(x: int) -> Any:
            return x / 20, -x / 20, x / -20, x / 1

        def div_int_var_and_float(x: float) -> Any:
            return x / 20.1, -x / 20.1, x / -20.1, x / 1.0

        def div_float_var_and_int(x: int) -> Any:
            return x / 20, -x / 20, x / -20, x / 1

        def div_float_var_and_float(x: float) -> Any:
            return x / 20.1, -x / 20.1, x / -20.1, x / 1.0

        test_funcs = [
            div_int_and_int_var,
            div_int_and_float_var,
            div_float_and_int_var,
            div_float_and_float_var,
            div_int_var_and_int,
            div_int_var_and_float,
            div_float_var_and_int,
            div_float_var_and_float,
        ]

        for f in test_funcs:
            print("[ConstFold] test func: ", f.__name__)
            ann = f.__annotations__.get("x")
            input_x = ann(10.1)
            tx_func = matx.script(f)
            self.assertEqual(f(input_x), tx_func(input_x))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
