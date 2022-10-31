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

import base64
import unittest
import matx
import random


class TestBase64(unittest.TestCase):

    def test_basic_usage(self):
        def b64encode(s: bytes) -> bytes:
            return base64.b64encode(s)

        def b64decode(s: bytes) -> bytes:
            return base64.b64decode(s)

        b64encode_op = matx.script(b64encode)
        b64decode_op = matx.script(b64decode)

        # 随机测试100次
        random.seed(201308)
        for _ in range(100):
            l = random.randint(30, 1200)
            ba = bytearray([random.randint(0, 255) for _ in range(l)])
            s = bytes(ba)
            py_encoded = b64encode(s)
            tx_encoded = b64encode_op(s)
            self.assertEqual(py_encoded, tx_encoded)

            py_decoded = b64decode(py_encoded)
            tx_decoded = b64decode_op(tx_encoded)
            self.assertEqual(py_decoded, tx_decoded)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
