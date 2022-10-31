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
import copy
from typing import List, Tuple
from typing import Any
from matx.runtime import _ffi_api
from matx.runtime.object_generic import to_runtime_object
from matx import FTList


def make_ft_list(seq=()):
    new_seqs = [to_runtime_object(x) for x in seq]
    return _ffi_api.FTList(*new_seqs)


class TestMatxListSorted(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, '../data/unicode_language.txt')) as f:
            self.unicode_content = f.read()
        return super().setUp()

    def test_python_list_sorted(self):
        @matx.script
        def sorted_list(it: List, key: Any = None, reverse: bool = False) -> List:
            return sorted(it, key, reverse)

        original: List = [1.0, 3.0, 2.2, 5, 4.3, 2.15]
        expected: List = sorted(original)
        original_copy: List = [1.0, 3.0, 2.2, 5, 4.3, 2.15]
        sorted_result: List = list(sorted_list(original_copy))
        self.assertListEqual(sorted_result, expected)
        self.assertListEqual(original_copy, original)  # make sure python_list did not change

    def test_python_list_sorted_reverse(self):
        @matx.script
        def sorted_list(it: List[float], key: Any = None, reverse: bool = False) -> List[float]:
            return sorted(it, key, reverse)

        original: List[float] = [1.0, 3.0, 2.2, 5, 4.3, 2.15]
        expected: List[float] = sorted(original, key=None, reverse=True)
        original_copy: List[float] = [1.0, 3.0, 2.2, 5, 4.3, 2.15]
        sorted_result: List[float] = list(sorted_list(original_copy, None, True))
        self.assertListEqual(sorted_result, expected)
        self.assertListEqual(original_copy, original)  # make sure python_list did not change

    def test_python_list_sorted_key(self):
        @matx.script
        def sorted_list(it: List[str], key: Any = None, reverse: bool = False) -> List[str]:
            return sorted(it, key, reverse)

        @matx.script
        def k(s: str) -> str:
            return s[1]

        original: List[str] = ["af", "be", "cd", "dc", "eb", "fa"]
        expected: List[str] = sorted(original, key=k)
        original_copy: List[str] = ["af", "be", "cd", "dc", "eb", "fa"]
        sorted_result: List[str] = list(sorted_list(original_copy, k))
        self.assertListEqual(sorted_result, expected)
        self.assertListEqual(original_copy, original)  # make sure python_list did not change

    def test_python_tuple_sorted_key(self):
        @matx.script
        def sorted_list(it: Tuple[str], key: Any = None, reverse: bool = False) -> List[str]:
            return sorted(it, key, reverse)

        @matx.script
        def k(s: str) -> str:
            return s[1]

        original: Tuple[str, ...] = ("af", "be", "cd", "dc", "eb", "fa")
        expected: List[str] = sorted(original, key=k)
        original_copy: Tuple[str, ...] = ("af", "be", "cd", "dc", "eb", "fa")
        sorted_result: List[str] = list(sorted_list(original_copy, k))
        self.assertListEqual(sorted_result, expected)
        self.assertTupleEqual(original_copy, original)  # make sure python_list did not change

    def test_python_tuple_sorted_reverse(self):
        @matx.script
        def sorted_list(it: Tuple[str], key: Any = None, reverse: bool = False) -> List:
            return sorted(it, key, reverse)

        original: Tuple = (1.0, 3.0, 2.2, 5, 4.3, 2.15)
        expected: List = sorted(original, key=None, reverse=True)
        original_copy: Tuple = (1.0, 3.0, 2.2, 5, 4.3, 2.15)
        sorted_result: List = list(sorted_list(original_copy, None, True))
        self.assertListEqual(sorted_result, expected)
        self.assertTupleEqual(original_copy, original)  # make sure python_list did not change

    def test_python_ftlist_sorted_reverse(self):
        @matx.script
        def sorted_list(it: matx.FTList, key: Any = None, reverse: bool = False) -> List[float]:
            return sorted(it, key, reverse)

        original: List[float] = [1.0, 3.0, 2.2, 5, 4.3, 2.15]
        expected: List[float] = sorted(original)
        original_copy: matx.runtime.List = make_ft_list(original)
        sorted_result: List[float] = list(sorted_list(original_copy))
        self.assertListEqual(sorted_result, expected)
        self.assertListEqual([original_copy[i] for i in range(
            len(original_copy))], original)  # make sure python_list did not change


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
