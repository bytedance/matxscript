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
from logging import fatal
from typing import List, Tuple, Dict
import unittest
import matx


class TestRegex(unittest.TestCase):

    def test_regex_compile(self):
        @matx.script
        def run_regex() -> Tuple[List, Tuple[List, Dict], Tuple[List, Dict], str]:
            regex = matx.Regex("name", False, False)
            tokens = regex.split("mynameisHE")
            match_0 = regex.match("mynameisHE")
            match_6 = regex.match("mynameisHE", 6)
            new_str = regex.replace("mynameisHE", "NAME")
            return tokens, match_0, match_6, new_str

        token, match_0, match_6, new_str = run_regex()
        assert len(token) == 2
        assert match_0 == matx.Tuple(matx.List(['name']), matx.Dict())
        assert match_6 == matx.Tuple(matx.List(), matx.Dict())
        assert new_str == "myNAMEisHE"

        @matx.script
        def run_regex_named_group() -> Tuple[List, Dict]:
            regex = matx.Regex("(?<first>.*) are (?<second>.*?) .*")
            matched_result = regex.match("Cats are smarter than dogs")
            return matched_result

        matched_result = run_regex_named_group()
        assert matched_result == matx.Tuple(matx.List(
            ['Cats are smarter than dogs', 'Cats', 'smarter']),
            matx.Dict({'first': 'Cats', 'second': 'smarter'}))

    def test_regex_annotate(self):
        @matx.script
        def make_regex() -> matx.Regex:
            regex = matx.Regex("name", False, False)
            return regex

        regex = make_regex()
        assert regex.match("mynameisHe") == matx.Tuple(matx.List(['name']), matx.Dict())

    def test_regex_for_archer(self):
        @matx.script
        def make_whitespace_regex() -> matx.Regex:
            regex = matx.Regex(r'[ \t\n\r\f\p{Zs}]', ignore_case=False, dotall=False)
            return regex

        @matx.script
        def make_control_regex() -> matx.Regex:
            regex = matx.Regex(r'[\u0000\ufffd\p{Cc}\p{Cf}\p{Mn}]', ignore_case=False, dotall=False)
            return regex

        ws_regex = make_whitespace_regex()
        ctrl_regex = make_control_regex()

        test_case = "hello\t \rgg\n\twor\ufffdld!\x00a\u200da"

        output = ws_regex.replace(test_case, " ")
        output = ctrl_regex.replace(output, "")
        print(output)
        self.assertEqual(output, "hello   gg  world!aa")

    def test_regex_type_convert(self):

        @matx.script
        def clean_string(ss: str) -> str:
            replaces = []
            replace_pair = [
                (r"[^A-Za-z0-9(),!?\'\`]", " "),
                (r"\'s", " \'s"),
                (r"\'ve", " \'ve"),
                (r"n\'t", " n\'t"),
                (r"\'re", " \'re"),
                (r"\'d", " \'d"),
                (r"\'ll", " \'ll"),
                (r",", " , "),
                (r"!", " ! "),
                (r"\(", " \\( "),
                (r"\)", " \\) "),
                (r"\?", " \\? "),
                (r"\s{2,}", " ")
            ]
            for tup in replace_pair:
                replaces.append((matx.Regex(tup[0], False, False), tup[1]))
            ss1 = ss
            for tup in replaces:
                ss1 = tup[0].replace(ss1, tup[1])
            return ss1.strip()

        assert clean_string('1,2,3') == '1 , 2 , 3'

    def test_special_replace(self):
        def replace_with_empty(s: bytes) -> bytes:
            r = matx.Regex(" ")
            return r.replace(s, b"")

        text = b"a b"
        py_ret = replace_with_empty(text)
        tx_ret = matx.script(replace_with_empty)(text)
        self.assertEqual(py_ret, tx_ret)

        def replace_with_0(s: bytes) -> bytes:
            r = matx.Regex(" ")
            return r.replace(s, b"\x00abc\x00")

        text = b"a b"
        py_ret = replace_with_0(text)
        tx_ret = matx.script(replace_with_0)(text)
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
