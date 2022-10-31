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
import unittest
import matx
from typing import List, Tuple

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestFile(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"

    def test_readline(self):
        def readlines_string(path: str) -> List:
            ret = []
            f = open(path, 'r', encoding='utf8')
            ret.append(f.readline())
            for line in f:
                ret.append(line)
            f.close()
            return ret

        def readlines_bytes(path: str) -> List:
            ret = []
            uf = open(path, 'rb')
            ret.append(uf.readline())
            for line in uf:
                ret.append(line)
            uf.close()
            return ret

        file_1 = self.data_path + "/input.utf8.txt"
        file_2 = self.data_path + "/input.utf8.txt.nonewline"

        scripted_output = matx.script(readlines_string)(file_1)
        builtin_output = readlines_string(file_1)
        self.assertEqual(builtin_output, scripted_output)

        scripted_output = matx.script(readlines_string)(file_2)
        builtin_output = readlines_string(file_2)
        self.assertEqual(builtin_output, scripted_output)

        scripted_output = matx.script(readlines_bytes)(file_1)
        builtin_output = readlines_bytes(file_1)
        self.assertEqual(builtin_output, scripted_output)

        scripted_output = matx.script(readlines_bytes)(file_2)
        builtin_output = readlines_bytes(file_2)
        self.assertEqual(builtin_output, scripted_output)

    def test_readlines(self):
        def readlines_string(path: str) -> List:
            f = open(path, 'r', encoding='utf8')
            ret = f.readlines()
            f.close()
            return ret

        def readlines_bytes(path: str) -> List:
            f = open(path, 'rb')
            ret = f.readlines()
            f.close()
            return ret

        path = self.data_path + "/input.utf8.txt"
        self.assertEqual(matx.script(readlines_string)(path), readlines_string(path))
        self.assertEqual(matx.script(readlines_bytes)(path), readlines_bytes(path))

    def test_multi_open(self):
        def share_open(path: str) -> matx.List:
            ret = matx.List()
            f1 = open(path, 'r', encoding='utf8')
            f2 = f1
            ret.append(f1.readline())
            ret.append(f2.readline())
            ret.append(f1.readline())
            f1.close()
            return ret

        path = self.data_path + "/input.utf8.txt.nonewline"
        scripted_output = matx.script(share_open)(path)
        builtin_output = share_open(path)
        self.assertEqual(builtin_output, scripted_output)

    def test_annotation(self):
        # TODO: Ann Iterable ?
        def readlines(f: matx.File) -> List:
            ret = []
            for line in f:
                ret.append(line)
            return ret

        def wrapper(path: str) -> List:
            f = open(path, 'r', encoding='utf8')
            return readlines(f)

        path = self.data_path + "/input.utf8.txt"
        ret = matx.script(wrapper)(path)
        builtin_output = wrapper(path)
        self.assertEqual(ret, builtin_output)

    def test_file_read(self):
        def file_read(loc: str) -> Tuple[str, str]:
            f = open(loc, 'r')
            head = f.read(12)
            data = f.read(145)
            for line in f:
                data += line
            f.close()
            return head, data

        path = self.data_path + "/input.utf8.txt.nonewline"
        py_ret = file_read(path)
        tx_ret = matx.script(file_read)(path)
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
