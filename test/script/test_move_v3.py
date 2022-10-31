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

from typing import List
import unittest
import matx


class TestAutoMoveV3(unittest.TestCase):
    def test_str_lower_and_concat(self):
        def my_str_lower_and_concat(query_tok: List[str]) -> List[str]:
            inputs: List[str] = []
            sep = " "
            for i, q_tok in enumerate(query_tok):
                if i > 0:
                    inputs.append(sep)
                tmp = q_tok.lower()
                inputs.append(tmp)

            return inputs

        examples = ["Hello", "World"]
        py_ret = my_str_lower_and_concat(examples)
        tx_ret = matx.script(my_str_lower_and_concat)(examples)
        self.assertEqual(tx_ret, py_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
