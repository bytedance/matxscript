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
from .. import _ffi
from .object import Object
from . import _ffi_api
from .object_generic import ObjectTypes, to_runtime_object

from typing import Union


@_ffi.register_object("Regex")
class Regex(Object):
    """Regular class implemented using pcre.

    Args:
        pattern (str): Str types. Regular expression pattern.
        ignore_case (bool): Booleans. Perform case-insensitive matching. The default is false
        dotall (bool): Booleans. "." matches any character at all, including the newline. The default is false
        extended (bool): Booleans. Most white space in the pattern (other than in a character class), and characters between a # outside a character class and the next newline, inclusive, are ignored. An escaping backslash can be used to include a white space or # character as part of the pattern. The default is false.
        anchored (bool): Booleans. Matches only at the beginning of the subject. The default is false.
        ucp (bool): Booleans. Sequences such as "\\d" and "\\w" use Unicode properties to determine character types, instead of recognizing only characters with codes less than 128 via a lookup table. The default is false.



    Examples:
        >>> import matx
        >>> regex = matx.Regex("(?<first>.*) are (?<second>.*?) .*")
        >>> regex
        Object(0x55c11322a200)
    """

    __hash__ = None

    def __init__(
            self,
            pattern,
            ignore_case=False,
            dotall=False,
            extended=False,
            anchored=False,
            ucp=True):
        self.__init_handle_by_constructor__(
            _ffi_api.Regex,
            to_runtime_object(pattern),
            ignore_case,
            dotall,
            extended,
            anchored,
            ucp)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def split(self, string: Union[str, bytes]):
        """Split a string by the occurrences of a pattern.

        Args:
            string (str|bytes): The source string.

        Returns:
            List[str|bytes]: A list containing the resulting substrings. If no match was found，returning a list containing only the source string, i.e. [input].

        Examples:
            >>> import matx
            >>> regex = matx.Regex("name")
            >>> tokens = regex.split("mynameisHE")
            >>> tokens
            ['my', 'isHE']
        """
        return _ffi_api.RegexSplit(self, string)

    def replace(self, string: Union[str, bytes], repl: Union[str, bytes]):
        """Return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in the input string by the replacement repl.

        Args:
            string (str|bytes): The source string.
            repl (str|bytes): The replacement string.

        Returns:
            str|bytes: The replaced string. If no match was found, returning the source string.

        Examples:
            >>> import matx
            >>> regex = matx.Regex("name")
            >>> new_str = regex.replace("mynameisHE", "NAME")
            >>> new_str
            myNAMEisHE
        """
        return _ffi_api.RegexReplace(self, string, repl)

    def match(self, string: Union[str, bytes], offset: int = 0):
        """Try to apply the pattern at the start of the string, returning a tuple containing the matched string. If grouping version of regular pattern is used, then the text of all groups are returned.

        Args:
            string (str|bytes): The source string.
            offset (int): Offset in the subject at which to start matching

        Returns:
            Tuple(List, Dict): The matched groups. The first element in the tuple is indexed groups. The second element in the tuple is named groups.

        Examples:
            >>> import matx
            >>> regex = matx.Regex("(?<first>.*) are (?<second>.*?) .*")
            >>> matched_result = regex.match("Cats are smarter than dogs")
            >>> matched_result[0]
            ['Cats are smarter than dogs', 'Cats', 'smarter']
            >>> matched_result[1]
            {'first': 'Cats', 'second': 'smarter'}
        """
        return _ffi_api.RegexMatch(self, string, offset)
