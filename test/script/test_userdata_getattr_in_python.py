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


class MyUserData:

    def __init__(self, a1: int) -> None:
        self.a: int = a1

    def get(self) -> int:
        return self.a


def make_my_user_data(i: int) -> MyUserData:
    return MyUserData(i)


class TestUserDataGetAttrInPython(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_getattr(self):
        py_ret = make_my_user_data(1).get()
        op = matx.script(make_my_user_data)
        data = op(1)
        tx_ret = data.get()
        self.assertEqual(py_ret, tx_ret)
        del data
        del op


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
