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
import os
os.environ['MATX_DEV_MODE'] = '1'

import matx
import unittest
from typing import Any


class TestErrorStack(unittest.TestCase):

    def test_index_error(self):

        @matx.script
        def err_fun(i: int) -> int:
            x = [1, 2, 3]
            return x[i]

        try:
            err_fun(3)
        except Exception as e:
            ex = str(e)
            print(ex)
            assert "1: err_fun" in ex
            assert "0: matxscript::runtime::List::get_item" in ex

    # backtrace can only be called once with precise location. mute the following 2 test first.
    def _test_key_error(self):

        @matx.script
        def key_error(s: str) -> int:
            x = {"a": 1, "b": 2}
            return x[s]

        key_error("c")

    def _test_type_error(self):

        def type_eraser(x: Any) -> Any:
            return x

        @matx.script
        def type_error() -> None:
            x = [1, 2, 3]
            y = type_eraser(x)
            print(y["c"])

        type_error()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
