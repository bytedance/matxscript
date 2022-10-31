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
from typing import Any


class TestOs(unittest.TestCase):

    def test_os_getenv(self):
        def getenv(k: str, d: Any = None) -> Any:
            if d is None:
                return os.getenv(k)
            return os.getenv(k, d)

        scripted_func = matx.script(getenv)
        os.environ['TEST_ENV'] = 'mytest'
        self.assertEqual(getenv('TEST_ENV'), 'mytest')
        self.assertEqual(getenv('TEST_ENV'), scripted_func('TEST_ENV'))

        self.assertEqual(getenv('NOT_EXISTS'), None)
        self.assertEqual(getenv('NOT_EXISTS'), scripted_func('NOT_EXISTS'))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
