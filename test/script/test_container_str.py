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
from typing import Any
from matx import FTList


class TestContainerStr(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, '../data/unicode_language.txt')) as f:
            self.unicode_content = f.read()
        return super().setUp()

    def test_str_lower(self):
        @matx.script
        def str_lower(s: str) -> str:
            return s.lower()

        self.assertEqual(str_lower(self.unicode_content), self.unicode_content.lower())

    def test_generic_lower(self):
        @matx.script
        def generic_lower(s: Any) -> Any:
            return s.lower()

        self.assertEqual(generic_lower(self.unicode_content), self.unicode_content.lower())

    def test_str_upper(self):
        @matx.script
        def str_upper(s: str) -> str:
            return s.upper()

        self.assertEqual(str_upper(self.unicode_content), self.unicode_content.upper())

    def test_generic_upper(self):
        @matx.script
        def generic_upper(s: Any) -> Any:
            return s.upper()

        self.assertEqual(generic_upper(self.unicode_content), self.unicode_content.upper())

    def test_str_isalpha(self):
        @matx.script
        def str_isalpha(s: str) -> bool:
            return s.isalpha()

        self.assertTrue(str_isalpha("HellO"))

    def test_generic_isalpha(self):
        @matx.script
        def generic_isalpha(s: Any) -> bool:
            return s.isalpha()

        self.assertTrue(generic_isalpha("HellO"))

    def test_str_isdigit(self):
        @matx.script
        def str_isdigit(s: str) -> bool:
            return s.isdigit()

        self.assertTrue(str_isdigit("123"))

    def test_generic_isdigit(self):
        @matx.script
        def generic_isdigit(s: Any) -> bool:
            return s.isdigit()

        self.assertTrue(generic_isdigit("123"))

    def test_list_slice_isdigit(self):
        @matx.script
        def list_slice_isdigit(s: matx.List) -> bool:
            return s[0].isdigit()

        self.assertTrue(list_slice_isdigit(matx.List(["123"])))

    def test_str_encode(self):
        @matx.script
        def str_encode(s: str) -> bytes:
            return s.encode()

        self.assertEqual(str_encode("\u624b\u5fc3"), "\u624b\u5fc3".encode())

    def test_generic_encode(self):
        @matx.script
        def generic_encode(s: Any) -> Any:
            return s.encode()

        self.assertEqual(generic_encode("\u624b\u5fc3"), "\u624b\u5fc3".encode())

    def test_str_decode(self):
        @matx.script
        def str_decode(s: bytes) -> str:
            return s.decode()

        self.assertEqual(str_decode("\u624b\u5fc3".encode()), "\u624b\u5fc3")

    def test_generic_decode(self):
        @matx.script
        def generic_decode(s: Any) -> Any:
            return s.decode()

        self.assertEqual(generic_decode("\u624b\u5fc3".encode()), "\u624b\u5fc3")

    def test_return_str(self):
        @matx.script
        def return_str() -> str:
            return "\' \t\n\v\f \""

        self.assertEqual(return_str(), "\' \t\n\v\f \"")

    def test_str_equal(self):
        @matx.script
        def str_equal(a: str, b: str) -> bool:
            return a == b

        self.assertTrue(str_equal("abc", "abc"))

    def test_bytes_equal(self):
        @matx.script
        def bytes_equal(a: bytes, b: bytes) -> bool:
            return a == b

        self.assertTrue(bytes_equal(b"abc", b"abc"))

    def test_generic_equal(self):
        @matx.script
        def generic_equal(a: Any, b: Any) -> bool:
            return a == b

        self.assertTrue(generic_equal("abc", "abc"))
        self.assertTrue(generic_equal(b"abc", b"abc"))
        self.assertFalse(generic_equal(b"", b"abc"))
        self.assertFalse(generic_equal("", "abc"))

    def test_str_add(self):
        def str_add(a: str, b: str) -> str:
            return a + b

        def str_imm_add() -> str:
            return '123' + 'hello'

        self.assertEqual(str_add('123', 'hello'), '123hello')
        self.assertEqual(str_imm_add(), '123hello')

    def test_generic_add(self):
        def generic_add(a: Any, b: Any) -> str:
            return a + b

        self.assertEqual(generic_add('123', 'hello'), '123hello')

    def test_str_find(self):
        @matx.script
        def str_find(tar: str) -> int:
            a = tar
            return a.find('\u54c8\u54c8')

        self.assertEqual(str_find('\u8fd9\u662fhhh,\u54c8\u54c8\u54c8,?'), 6)
        self.assertEqual(str_find('not found---\u54c8---'), -1)

        @matx.script
        def str_find_one_arg(tar: str) -> int:
            a = tar
            return a.find('\u54c8\u54c8', 7)

        self.assertEqual(str_find_one_arg('\u8fd9\u662fhhh,\u54c8\u54c8\u54c8,?,---\u54c8---'), 7)

        @matx.script
        def str_find_more_arg(tar: str) -> int:
            a = tar
            return a.find('\u54c8\u54c8', 7, 9)

        self.assertEqual(str_find_more_arg('\u8fd9\u662fhhh,\u54c8\u54c8\u54c8,?,---\u54c8---'), 7)

    def test_str_compare(self):
        @matx.script
        def str_lte(s1: str, s2: str) -> bool:
            return s1 <= s2

        self.assertTrue(str_lte('a', 'b'))
        self.assertFalse(str_lte('bbb', 'aaa'))

    def test_chaining_comparision(self):
        @matx.script
        def chaining_lt(a: str, b: str, c: str) -> bool:
            return a < b < c

        self.assertTrue(chaining_lt('1', '2', '3'))

    def test_split(self):
        @matx.script
        def str_split(s: str, sep: Any, maxsplit: int) -> matx.List:
            if sep is None and maxsplit == -1:
                return s.split()
            elif maxsplit == -1:
                return s.split(sep)
            else:
                return s.split(sep, maxsplit)

        @matx.script
        def str_split_generic(slist: matx.List, sep: str, maxsplit: int) -> matx.List:
            ret = matx.List()
            for s in slist:
                ret.append(s.split(sep, maxsplit))
            return ret

        # From issue #125
        @matx.script
        def str_split_generic_iter_get(batch_string_list: List[str]) -> List[List[str]]:
            output = matx.List()
            output.reserve(len(batch_string_list))
            for ss in batch_string_list:
                token_list = matx.List()
                token_list.reserve(len(ss))
                for token in ss.split(' '):
                    token_list.append(token)
                output.append(token_list)

            return output

        @matx.script
        def str_ft_split() -> None:
            a = b'hello world'
            ret1: FTList[bytes] = a.split(b' ')
            ret2: FTList[Any] = a.split(b' ')
            assert ret1 == ret2, 'ret1 != ret2'
            print(ret1)

            b = 'hello world'
            ret3: FTList[str] = b.split(' ')
            ret4: FTList[Any] = b.split(' ')
            assert ret3 == ret4, 'ret3 != ret4'
            print(ret3)

        def invalid_str_ft_split() -> None:
            c: object = 'hello world'
            ret: FTList[Any] = c.split(' ')

        self.assertEqual(str_split('1 2\t3\n4', None, -1), matx.List('1 2\t3\n4'.split()))
        self.assertEqual(str_split('1 2 3\n4', ' ', -1), matx.List('1 2 3\n4'.split(' ')))
        self.assertEqual(
            str_split(
                '1 2 3\n4', None, 2), matx.List(
                '1 2 3\n4'.split(
                    maxsplit=2)))
        self.assertEqual(
            str_split('1ab2ab3ab4', 'ab', 1),
            matx.List('1ab2ab3ab4'.split('ab', 1))
        )
        self.assertEqual(
            str_split_generic(['1 2 3', '4 5'], ' ', 1),
            matx.List([['1', '2 3'], ['4', '5']])
        )
        self.assertEqual(str_split_generic_iter_get(['z f b']),
                         matx.List([['z', 'f', 'b']]))

        str_ft_split()
        with self.assertRaises(Exception):
            matx.script(invalid_str_ft_split)

    def test_join(self):
        @matx.script
        def str_join(s: str, __iterable: Any) -> str:
            return s.join(__iterable)

        @matx.script
        def str_join_list(s: str, __list: matx.List) -> str:
            return s.join(__list)

        @matx.script
        def object_join(s: str, __iterable: Any) -> str:
            return s.join(__iterable)

        @matx.script
        def object_join_list(s: Any, __list: matx.List) -> str:
            return s.join(__list)

        @matx.script
        def str_join_ftlist(s: str, __list: FTList[str]) -> str:
            return s.join(__list)

        @matx.script
        def gen_ftlist() -> FTList[str]:
            return ['1', '23', '4']

        self.assertEqual(str_join('', ['1', '23', '4']), '1234')
        self.assertEqual(object_join('', ['1', '23', '4']), '1234')
        self.assertEqual(str_join('||', ['1', '23', '4']), '1||23||4')
        self.assertEqual(object_join('||', ['1', '23', '4']), '1||23||4')
        self.assertEqual(str_join(',', []), '')
        self.assertEqual(object_join(',', []), '')

        self.assertEqual(str_join_list('||', ['1', '23', '4']), '1||23||4')
        self.assertEqual(object_join_list('||', ['1', '23', '4']), '1||23||4')

        ft_list: FTList[str] = gen_ftlist()
        self.assertEqual(str_join_ftlist('||', ft_list), '1||23||4')
        self.assertEqual(str_join('||', ft_list), '1||23||4')
        # TODO(maxiandi): fix destructor order
        del ft_list

    def test_replace(self):
        @matx.script
        def str_replace(s: str, o: str, n: str, c: int) -> str:
            if c == -1:
                return s.replace(o, n)
            else:
                return s.replace(o, n, c)

        @matx.script
        def bytes_replace(s: bytes, o: bytes, n: bytes, c: int) -> bytes:
            if c == -1:
                return s.replace(o, n)
            else:
                return s.replace(o, n, c)

        @matx.script
        def generic_replace(s: Any, o: Any, n: Any, c: int) -> Any:
            if c == -1:
                return s.replace(o, n)
            else:
                return s.replace(o, n, c)

        self.assertEqual(str_replace('aabbcc', 'a', 'b', -1), 'bbbbcc')
        self.assertEqual(str_replace('aabbcc', 'a', 'b', 1), 'babbcc')

        self.assertEqual(bytes_replace(b'aabbcc', b'a', b'b', -1), b'bbbbcc')
        self.assertEqual(bytes_replace(b'aabbcc', b'a', b'b', 1), b'babbcc')

        self.assertEqual(generic_replace('aabbcc', 'a', 'b', -1), 'bbbbcc')
        self.assertEqual(generic_replace('aabbcc', 'a', 'b', 1), 'babbcc')

        self.assertEqual(generic_replace(b'aabbcc', b'a', b'b', -1), b'bbbbcc')
        self.assertEqual(generic_replace(b'aabbcc', b'a', b'b', 1), b'babbcc')

    def test_startswith(self):
        @matx.script
        def str_startswith(s: str, prefix: str, start: int, end: int) -> bool:
            if start == -100 and end == -100:
                return s.startswith(prefix)
            elif end == -100:
                return s.startswith(prefix, start)
            else:
                return s.startswith(prefix, start, end)

        @matx.script
        def str_startswith_tup(s: str) -> bool:
            prefixes: Tuple[str, str] = ('hh', 'uu')
            return s.startswith(prefixes)

        def str_startswith_tup_except(s: str) -> bool:
            prefixes: Tuple[bytes, bytes] = (b'hh', b'uu')
            return s.startswith(prefixes)

        @matx.script
        def bytes_startswith(s: bytes, prefix: bytes, start: int, end: int) -> bool:
            if start == -100 and end == -100:
                return s.startswith(prefix)
            elif end == -100:
                return s.startswith(prefix, start)
            else:
                return s.startswith(prefix, start, end)

        @matx.script
        def bytes_startswith_tup(s: bytes) -> bool:
            prefixes: Tuple[bytes, bytes] = (b'hh', b'uu')
            return s.startswith(prefixes)

        def bytes_startswith_tup_except(s: bytes) -> bool:
            prefixes: Tuple[str, str] = ('hh', 'uu')
            return s.startswith(prefixes)

        @matx.script
        def generic_startswith(
                s: Any,
                prefix_or_prefixes: Any,
                start: Any,
                end: Any) -> bool:
            if start == -100 and end == -100:
                return s.startswith(prefix_or_prefixes)
            elif end == -100:
                return s.startswith(prefix_or_prefixes, start)
            else:
                return s.startswith(prefix_or_prefixes, start, end)

        self.assertEqual(str_startswith('01234567890', '01', -100, -100), True)
        self.assertEqual(str_startswith('01234567890', '12', 1, 3), True)
        self.assertEqual(str_startswith('01234567890', '89', -3, -100), True)
        self.assertEqual(str_startswith('01234567890', '01', 1, -100), False)
        self.assertEqual(str_startswith('01234567890', '90', 9, 10), False)
        self.assertEqual(str_startswith_tup('hhabc'), True)
        self.assertEqual(str_startswith_tup('uuabc'), True)
        self.assertEqual(str_startswith_tup('abc'), False)
        with self.assertRaises(Exception):
            matx.script(str_startswith_tup_except)('hhabc')

        self.assertEqual(bytes_startswith(b'01234567890', b'01', -100, -100), True)
        self.assertEqual(bytes_startswith(b'01234567890', b'12', 1, 3), True)
        self.assertEqual(bytes_startswith(b'01234567890', b'89', -3, -100), True)
        self.assertEqual(bytes_startswith(b'01234567890', b'01', 1, -100), False)
        self.assertEqual(bytes_startswith(b'01234567890', b'90', 9, 10), False)
        self.assertEqual(bytes_startswith_tup(b'hhabc'), True)
        self.assertEqual(bytes_startswith_tup(b'uuabc'), True)
        self.assertEqual(bytes_startswith_tup(b'abc'), False)
        with self.assertRaises(Exception):
            matx.script(bytes_startswith_tup_except)(b'hhabc')

        self.assertEqual(generic_startswith('01234567890', '01', -100, -100), True)
        self.assertEqual(generic_startswith('01234567890', '12', 1, 3), True)
        self.assertEqual(generic_startswith(b'01234567890', b'89', -3, -100), True)
        self.assertEqual(generic_startswith('01234567890', '01', 1, -100), False)
        self.assertEqual(generic_startswith('01234567890', '90', 9, 10), False)
        self.assertEqual(generic_startswith('01234567890', ('23', '01'), 2, 4), True)

    def test_endswith(self):
        @matx.script
        def str_endswith(s: str, suffix: str, start: int, end: int) -> bool:
            if start == -100 and end == -100:
                return s.endswith(suffix)
            elif end == -100:
                return s.endswith(suffix, start)
            else:
                return s.endswith(suffix, start, end)

        @matx.script
        def str_endswith_tup(s: str) -> bool:
            suffixes: Tuple[str, str] = ('hh', 'uu')
            return s.endswith(suffixes)

        @matx.script
        def bytes_endswith(s: bytes, suffix: bytes, start: int, end: int) -> bool:
            if start == -100 and end == -100:
                return s.endswith(suffix)
            elif end == -100:
                return s.endswith(suffix, start)
            else:
                return s.endswith(suffix, start, end)

        @matx.script
        def bytes_endswith_tup(s: bytes) -> bool:
            suffixes: Tuple[bytes, bytes] = (b'hh', b'uu')
            return s.endswith(suffixes)

        @matx.script
        def generic_endswith(
                s: Any,
                suffix_or_suffixes: Any,
                start: Any,
                end: Any) -> bool:
            if start == -100 and end == -100:
                return s.endswith(suffix_or_suffixes)
            elif end == -100:
                return s.endswith(suffix_or_suffixes, start)
            else:
                return s.endswith(suffix_or_suffixes, start, end)

        self.assertEqual(str_endswith('01234567890', '90', -100, -100), True)
        self.assertEqual(str_endswith('01234567890', '89', 0, 10), True)
        self.assertEqual(str_endswith('01234567890', '678', 0, -2), True)
        self.assertEqual(str_endswith('01234567890', '90', 0, 10), False)
        self.assertEqual(str_endswith('01234567890', '89', 9, 10), False)
        self.assertEqual(str_endswith_tup('abchh'), True)
        self.assertEqual(str_endswith_tup('abcuu'), True)
        self.assertEqual(str_endswith_tup('abc'), False)

        self.assertEqual(bytes_endswith(b'01234567890', b'90', -100, -100), True)
        self.assertEqual(bytes_endswith(b'01234567890', b'89', 0, 10), True)
        self.assertEqual(bytes_endswith(b'01234567890', b'678', 0, -2), True)
        self.assertEqual(bytes_endswith(b'01234567890', b'90', 0, 10), False)
        self.assertEqual(bytes_endswith(b'01234567890', b'89', 9, 10), False)
        self.assertEqual(bytes_endswith_tup(b'abchh'), True)
        self.assertEqual(bytes_endswith_tup(b'abcuu'), True)
        self.assertEqual(bytes_endswith_tup(b'abc'), False)

        self.assertEqual(generic_endswith('01234567890', '90', -100, -100), True)
        self.assertEqual(generic_endswith('01234567890', '89', 0, 10), True)
        self.assertEqual(generic_endswith(b'01234567890', b'678', 0, -2), True)
        self.assertEqual(generic_endswith('01234567890', '90', 0, 10), False)
        self.assertEqual(generic_endswith('01234567890', '89', 9, 10), False)
        self.assertEqual(generic_endswith('01234567890', ('89', '90'), -100, -100), True)

    def test_rstrip(self):
        @matx.script
        def str_rstrip(s: str, __chars: str) -> str:
            if len(__chars) == 0:
                return s.rstrip()
            else:
                return s.rstrip(__chars)

        @matx.script
        def bytes_rstrip(s: bytes, __chars: bytes) -> bytes:
            if len(__chars) == 0:
                return s.rstrip()
            else:
                return s.rstrip(__chars)

        @matx.script
        def generic_rstrip(s: Any, __chars: Any) -> Any:
            if len(__chars) == 0:
                return s.rstrip()
            else:
                return s.rstrip(__chars)

        self.assertEqual(str_rstrip('abc  \t', ''), 'abc')
        self.assertEqual(str_rstrip('abcde', 'de'), 'abc')
        self.assertEqual(str_rstrip('ab', 'ba'), '')

        self.assertEqual(bytes_rstrip(b'abc  \t', b''), b'abc')
        self.assertEqual(bytes_rstrip(b'abcde', b'de'), b'abc')
        self.assertEqual(bytes_rstrip(b'ab', b'ba'), b'')

        self.assertEqual(generic_rstrip('abc  \t', ''), 'abc')
        self.assertEqual(generic_rstrip(b'abcde', b'de'), b'abc')
        self.assertEqual(generic_rstrip('ab', 'ba'), '')

        self.assertEqual(str_rstrip('abc\xa0', ''), 'abc')

    def test_lstrip(self):
        @matx.script
        def str_lstrip(s: str, __chars: str) -> str:
            if len(__chars) == 0:
                return s.lstrip()
            else:
                return s.lstrip(__chars)

        @matx.script
        def bytes_lstrip(s: bytes, __chars: bytes) -> bytes:
            if len(__chars) == 0:
                return s.lstrip()
            else:
                return s.lstrip(__chars)

        @matx.script
        def generic_lstrip(s: Any, __chars: Any) -> Any:
            if len(__chars) == 0:
                return s.lstrip()
            else:
                return s.lstrip(__chars)

        self.assertEqual(str_lstrip('  \tabc', ''), 'abc')
        self.assertEqual(str_lstrip('abcde', 'ba'), 'cde')
        self.assertEqual(str_lstrip('ab', 'ba'), '')

        self.assertEqual(bytes_lstrip(b'  \tabc', b''), b'abc')
        self.assertEqual(bytes_lstrip(b'abcde', b'ba'), b'cde')
        self.assertEqual(bytes_lstrip(b'ab', b'ba'), b'')

        self.assertEqual(generic_lstrip('  \tabc', ''), 'abc')
        self.assertEqual(generic_lstrip(b'abcde', b'ba'), b'cde')
        self.assertEqual(generic_lstrip('ab', 'ba'), '')

    def test_strip(self):
        @matx.script
        def str_strip(s: str, __chars: str) -> str:
            if len(__chars) == 0:
                return s.strip()
            else:
                return s.strip(__chars)

        @matx.script
        def bytes_strip(s: bytes, __chars: bytes) -> bytes:
            if len(__chars) == 0:
                return s.strip()
            else:
                return s.strip(__chars)

        @matx.script
        def generic_strip(s: Any, __chars: Any) -> Any:
            if len(__chars) == 0:
                return s.strip()
            else:
                return s.strip(__chars)

        self.assertEqual(str_strip('  abc  \t', ''), 'abc')
        self.assertEqual(str_strip('abcde', 'dae'), 'bc')
        self.assertEqual(str_strip('ab', 'ba'), '')

        self.assertEqual(bytes_strip(b'  abc  \t', b''), b'abc')
        self.assertEqual(bytes_strip(b'abcde', b'dae'), b'bc')
        self.assertEqual(bytes_strip(b'ab', b'ba'), b'')

        self.assertEqual(generic_strip('  abc  \t', ''), 'abc')
        self.assertEqual(generic_strip(b'abcde', b'dae'), b'bc')
        self.assertEqual(generic_strip('ab', 'ba'), '')

    def test_count(self):
        @matx.script
        def str_count(s: str, x: str, __start: int, __end: int) -> int:
            if __start == -100 and __end == -100:
                return s.count(x)
            elif __end == -100:
                return s.count(x, __start)
            return s.count(x, __start, __end)

        @matx.script
        def bytes_count(s: bytes, x: bytes, __start: int, __end: int) -> int:
            if __start == -100 and __end == -100:
                return s.count(x)
            elif __end == -100:
                return s.count(x, __start)
            return s.count(x, __start, __end)

        @matx.script
        def generic_count(s: Any, x: Any, __start: Any, __end: Any) -> int:
            if __start == -100 and __end == -100:
                return s.count(x)
            elif __end == -100:
                return s.count(x, __start)
            return s.count(x, __start, __end)

        self.assertEqual(str_count("001122001122", "0", -100, -100), 4)
        self.assertEqual(str_count("001122001122", "0", 1, 4), 1)
        self.assertEqual(str_count("001122001122", "01", -100, -100), 2)
        self.assertEqual(str_count("001122001122", "01", 3, 12), 1)

        self.assertEqual(bytes_count(b"001122001122", b"0", -100, -100), 4)
        self.assertEqual(bytes_count(b"001122001122", b"0", 1, 4), 1)
        self.assertEqual(bytes_count(b"001122001122", b"01", -100, -100), 2)
        self.assertEqual(bytes_count(b"001122001122", b"01", 3, 12), 1)

        self.assertEqual(generic_count("001122001122", "0", -100, -100), 4)
        self.assertEqual(generic_count("001122001122", "0", 1, 4), 1)
        self.assertEqual(generic_count("001122001122", "01", -100, -100), 2)
        self.assertEqual(generic_count("001122001122", "01", 3, 12), 1)

    def test_ord(self):
        def str_ord(s: str) -> int:
            return ord(s)

        def bytes_ord(s: bytes) -> int:
            return ord(s)

        s = 'a'
        bs = b'a'
        ret = ord(s)
        matx_ret = matx.script(str_ord)(s)
        bret = ord(bs)
        bmatx_ret = matx.script(bytes_ord)(bs)
        self.assertEqual(ret, matx_ret)
        self.assertEqual(bret, bmatx_ret)

    def test_chr(self):
        def str_chr(i: int) -> str:
            return chr(i)

        scripted_func = matx.script(str_chr)
        for i in range(0x9FEF):
            self.assertEqual(scripted_func(i), str_chr(i))

    def test_format(self):
        def str_format() -> matx.List:
            ret = matx.List()
            s = 'abc'
            l = matx.List(["aaa", b"bbb"])
            ret.append('{{}}123')
            ret.append('11{}22')
            ret.append('{{}}123'.format(123))
            ret.append('11{}22{}'.format(s, l))
            ret.append('11{}22'.format(s, 255))
            ret.append('{{{}}}123'.format(123))
            ret.append('{{'.format())
            ret.append('}}'.format())
            ret.append('{}}}'.format(34))
            return ret

        def str_illegal_format1() -> str:
            return '{{{'.format()

        def str_illegal_format2() -> str:
            return '}}}'.format()

        def str_illegal_format3() -> str:
            return '{}'.format()

        self.assertEqual(str_format(), matx.script(str_format)())
        self.assertRaises(Exception, matx.script(str_illegal_format1))
        self.assertRaises(Exception, matx.script(str_illegal_format2))
        self.assertRaises(Exception, matx.script(str_illegal_format3))

        def generic_format(s: Any, param: Any) -> str:
            return s.format(param)

        self.assertEqual(generic_format('p: {}', matx.Dict({1: 2})),
                         matx.script(generic_format)('p: {}', matx.Dict({1: 2})))

    def test_repeat(self):
        @matx.script
        def str_repeat(s: str, t: int) -> str:
            return s * t

        @matx.script
        def bytes_repeat(b: bytes, t: int) -> bytes:
            return b * t

        @matx.script
        def generic_repeat(o: Any, t: int) -> Any:
            return o * t

        t = 3
        s = 'abc'
        b = b'abc'

        self.assertEqual(str_repeat(s, t), 'abcabcabc')
        self.assertEqual(bytes_repeat(b, t), b'abcabcabc')
        self.assertEqual(generic_repeat(s, t), 'abcabcabc')
        self.assertEqual(generic_repeat(b, t), b'abcabcabc')

    def test_str_contains(self):
        @matx.script
        def str_contain_specific(q: str, c: str) -> int:
            if c in q:
                return 1
            return 0

        @matx.script
        def str_contain_generic(q: str, c: Any) -> int:
            if c in q:
                return 1
            return 0

        self.assertEqual(1, str_contain_specific("abc", "bc"))
        self.assertEqual(1, str_contain_generic("abc", "bc"))

    def test_bytes_contains(self):
        @matx.script
        def bytes_contain_specific(q: bytes, c: bytes) -> int:
            if c in q:
                return 1
            return 0

        @matx.script
        def bytes_contain_generic(q: bytes, c: Any) -> int:
            if c in q:
                return 1
            return 0

        self.assertEqual(1, bytes_contain_specific(b"abc", b"bc"))
        self.assertEqual(1, bytes_contain_generic(b"abc", b"bc"))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
