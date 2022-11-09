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
from typing import Callable, List, Any, AnyStr

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestBuiltinTextOps(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"

    def test_wordpiece_tokenizer(self):
        vocab_path = self.data_path + os.sep + "vocab.txt"
        tokenizer = matx.text.WordPieceTokenizer(
            vocab_path=vocab_path,
            lookup_id=False,
            subwords_prefix="",
        )

        example = ["hello", "world", "helloworld", "kkk"]
        expect = ["hello", "world", "hello", "world", "[UNK]"]
        tx_ret1 = tokenizer(example)
        print(tx_ret1)
        self.assertEqual(tx_ret1, expect)

        tokenizer_op = matx.script(matx.text.WordPieceTokenizer)(
            vocab_path=vocab_path,
            lookup_id=False,
            subwords_prefix="",
        )

        tx_ret2 = tokenizer_op(example)
        print(tx_ret2)
        self.assertEqual(tx_ret2, expect)

        class MyTokenizer:
            def __init__(self):
                self.op: Callable = matx.text.WordPieceTokenizer(
                    vocab_path=vocab_path,
                    lookup_id=False,
                    subwords_prefix="",
                )

            def __call__(self, a: List[AnyStr]) -> Any:
                return self.op(a)

        tx_ret3 = matx.script(MyTokenizer)()(example)
        print(tx_ret3)
        self.assertEqual(tx_ret3, expect)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
