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

from typing import Tuple, List
from typing import Any
import os
import unittest
import matx


class TestContainerTrie(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.index_data = {}
        self.index_data["hello"] = 1
        self.index_data["hello world"] = 2
        return super().setUp()

    def test_runtime_build_trie(self):
        trie1 = matx.Trie()
        trie2 = matx.Trie(self.index_data)
        print(trie2.prefix_search("hello"))
        trie1.update("hello", 1)
        trie1.update("hello world", 2)
        self.assertEqual((5, 1), trie1.prefix_search("hello"))
        self.assertEqual((5, 1), trie1.prefix_search(b"hello"))

    def test_constructor(self):
        @matx.script
        def test_trie_constructor() -> matx.Trie:
            trie1 = matx.Trie()
            return trie1

        @matx.script
        def test_trie_constructor_with_args() -> matx.Trie:
            index_data = matx.Dict()
            index_data["hello"] = 1
            index_data["hello world"] = 2
            trie2 = matx.Trie(index_data)
            return trie2

        result = test_trie_constructor()
        self.assertTrue(isinstance(result, matx.Trie))
        self.assertEqual((0, -1), result.prefix_search("hello"))
        result = test_trie_constructor_with_args()
        self.assertTrue(isinstance(result, matx.Trie))
        self.assertEqual((5, 1), result.prefix_search("hello"))

    def test_update(self):
        @matx.script
        def test_trie_update() -> matx.Trie:
            trie1 = matx.Trie()
            trie1.update("hello", 1)
            trie1.update("hello world", 2)
            return trie1

        @matx.script
        def test_trie_generic_update(trie1: Any) -> Any:
            trie1.update("hello", 1)
            trie1.update("hello world", 2)
            return trie1

        result = test_trie_update()
        self.assertTrue(isinstance(result, matx.Trie))
        self.assertEqual((5, 1), result.prefix_search("hello"))
        result = test_trie_generic_update(matx.Trie())
        self.assertTrue(isinstance(result, matx.Trie))
        self.assertEqual((5, 1), result.prefix_search("hello"))

    def test_prefix_search(self):
        @matx.script
        def test_trie_prefix_search() -> Any:
            trie1 = matx.Trie()
            trie1.update("hello", 1)
            trie1.update("hello world", 2)
            # return trie1.prefix_search("hello")
            mblen, idx = trie1.prefix_search("hello")
            return mblen, idx

        @matx.script
        def test_trie_generic_prefix_search(trie1: Any) -> Tuple[int, int]:
            trie1.update("hello", 1)
            trie1.update("hello world", 2)
            # return trie1.prefix_search("hello")
            mblen, idx = trie1.prefix_search("hello")
            return mblen, idx

        mblen, idx = test_trie_prefix_search()
        self.assertEqual((5, 1), (mblen, idx))
        mblen, idx = test_trie_generic_prefix_search(matx.Trie())
        self.assertEqual((5, 1), (mblen, idx))

    def test_prefix_search_all(self):
        def test_trie_prefix_search_all(trie: matx.Trie, arg: Any) -> List:
            return trie.prefix_search_all(arg)

        def test_trie_generic_prefix_search_all(trie: Any, arg: Any) -> List:
            return trie.prefix_search_all(arg)

        op = matx.script(test_trie_prefix_search_all)
        generic_op = matx.script(test_trie_generic_prefix_search_all)

        trie = matx.Trie({'hello': 1, 'hello world': 2, 'hello hello world': 3})

        def check(arg, expected):
            ret = test_trie_prefix_search_all(trie, arg)
            s = set((length, idx) for length, idx in ret)
            print(s)
            self.assertEqual(s, expected)

            ret = op(trie, arg)
            s = set((length, idx) for length, idx in ret)
            print(s)
            self.assertEqual(s, expected)

            ret = generic_op(trie, arg)
            s = set((length, idx) for length, idx in ret)
            print(s)
            self.assertEqual(s, expected)

        check("hello hello world", set([(5, 1), (17, 3)]))
        check(b"hello hello world", set([(5, 1), (17, 3)]))

    def test_save_load(self):
        import os

        def rm_file(file_path):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

        def test_trie_specific_save(trie: matx.Trie, file_path: str) -> int:
            return trie.save(file_path)

        def test_trie_generic_save(trie: Any, file_path: str) -> int:
            return trie.save(file_path)

        def test_trie_specific_load(trie: matx.Trie, file_path: str) -> int:
            return trie.load(file_path)

        def test_trie_generic_load(trie: Any, file_path: str) -> int:
            return trie.load(file_path)

        specific_save_op = matx.script(test_trie_specific_save)
        generic_save_op = matx.script(test_trie_generic_save)
        specific_load_op = matx.script(test_trie_specific_load)
        generic_load_op = matx.script(test_trie_generic_load)

        file_path = "trie_save_load.da"
        trie = matx.Trie({'hello': 1, 'hello world': 2, 'hello hello world': 3})
        rm_file(file_path)
        self.assertEqual(0, test_trie_specific_save(trie, file_path))
        rm_file(file_path)
        self.assertEqual(0, specific_save_op(trie, file_path))
        rm_file(file_path)
        self.assertEqual(0, generic_save_op(trie, file_path))

        load_trie = matx.Trie()
        specific_load_trie = matx.Trie()
        generic_load_trie = matx.Trie()
        self.assertEqual(0, test_trie_specific_load(load_trie, file_path))
        self.assertEqual(0, specific_load_op(specific_load_trie, file_path))
        self.assertEqual(0, generic_load_op(generic_load_trie, file_path))

        def check(trie, arg, expected):
            ret = trie.prefix_search_all(arg)
            s = set((length, idx) for length, idx in ret)
            self.assertEqual(s, expected)

        check(load_trie, "hello hello world", {(5, 1), (17, 3)})
        check(specific_load_trie, "hello hello world", {(5, 1), (17, 3)})
        check(generic_load_trie, "hello hello world", {(5, 1), (17, 3)})
        os.remove(file_path)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
