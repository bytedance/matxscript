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

from io import UnsupportedOperation
from .. import _ffi
from .object import Object
from . import _ffi_api
from .object_generic import to_runtime_object
from .container import Dict


@_ffi.register_object("runtime.Trie")
class Trie(Object):
    """Double Array Trie Object

    Args:
        dic (Dict[str, int]):
            The key is word and the value is id

    Examples:
        >>> import matx
        >>> tree = matx.Trie({"hello": 1, "hello w": 2, "good": 3})
        >>> tree
        Trie(addr: 0x5601feb2aca0)
        >>> tree = matx.Trie()
        >>> tree
        Trie(addr: 0x5601feba5b90)
    """

    def __init__(self, dic=None):
        if dic is None:
            self.__init_handle_by_constructor__(_ffi_api.Trie)
        else:
            assert isinstance(dic, (dict, Dict))
            dic = to_runtime_object(dic)
            self.__init_handle_by_constructor__(_ffi_api.Trie, dic)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def prefix_search(self, w, pos=0):
        """Find the longest substring of w[pos:] in the trie tree

        Args:
            w (str): The input word
            pos (int, optional): The start position

        Returns:
            Tuple(int, int): Return the length and id of the matched substring.
                If not found, return (0, -1)

        Examples:
            >>> import matx
            >>> tree = matx.Trie({"hello": 1, "hello w": 2, "good": 3})
            >>> tree.prefix_search("hello world")
            (7, 2)
            >>> tree.prefix_search("hellf")
            (0, -1)
        """
        w = to_runtime_object(w)
        mlen, index = _ffi_api.Trie_PrefixSearch(self, w, pos)
        return mlen, index

    def update(self, w, index=-1):
        """Insert a word and corresponding id into the trie tree

        Args:
            w (str): The input word
            index (int, optional): id, -1 for default

        Examples:
            >>> import matx
            >>> tree = matx.Trie()
            >>> tree.update("hello", 1)
            >>> tree.update("hello w", 2)
            >>> tree.update("good", 3)
            >>> tree.prefix_search("hello world")
            (7, 2)
            >>> tree.prefix_search("hellf")
            (0, -1)
            >>> tree.update("hell", 10)
            >>> tree.prefix_search("hellf")
            (4, 10)
        """
        w = to_runtime_object(w)
        _ffi_api.Trie_Update(self, w, index)

    def prefix_search_all(self, w, pos=0):
        """Find all substring of w[pos:] in the trie tree

        Args:
            w (str): The input word
            pos (int, optional): The start position

        Returns:
            List[Tuple(int, int)]: Return a list of the length and id of the matched substring.
                If not found, return []

        Examples:
            >>> import matx
            >>> trie = matx.Trie({'hello': 1, 'hello world': 2, 'hello hello world': 3})
            >>> trie.prefix_search_all("hello hello world")
            [(5, 1), (17, 3)]
        """
        return _ffi_api.Trie_PrefixSearchAll(self, w, pos)

    def save(self, file_path: str):
        return _ffi_api.Trie_Save(self, file_path)

    def load(self, file_path: str):
        return _ffi_api.Trie_Load(self, file_path)
