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
import msgpack

import matx


class TestMsgpack(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_runtime_msgpack(self):
        py_data = [None, 1, 1.1, 'hello', b'hi', [0, "hello"], {"h": 1}]
        tx_bytes = matx.msgpack_dumps(py_data)
        msgpack_bytes = msgpack.dumps(py_data)
        self.assertEqual(tx_bytes, msgpack_bytes)

        tx_data = matx.msgpack_loads(tx_bytes)
        msgpack_data = msgpack.loads(tx_bytes)

        self.assertEqual(tx_data, msgpack_data)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
