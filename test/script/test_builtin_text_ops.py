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
        tx_ret1 = tokenizer.tokenize(example)
        print(tx_ret1)
        self.assertEqual(tx_ret1, expect)

        tokenizer_op = matx.script(matx.text.WordPieceTokenizer)(
            vocab_path=vocab_path,
            lookup_id=False,
            subwords_prefix="",
        )

        tx_ret2 = tokenizer_op.tokenize(example)
        print(tx_ret2)
        self.assertEqual(tx_ret2, expect)

        class MyTokenizer:
            def __init__(self):
                self.op: matx.text.WordPieceTokenizer = matx.text.WordPieceTokenizer(
                    vocab_path=vocab_path,
                    lookup_id=False,
                    subwords_prefix="",
                )

            def __call__(self, a: List[AnyStr]) -> Any:
                return self.op.tokenize(a)

        tx_ret3 = matx.script(MyTokenizer)()(example)
        print(tx_ret3)
        self.assertEqual(tx_ret3, expect)

    def test_jieba(self):
        test_content = "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。"
        jieba = matx.text.Jieba()
        print(jieba.cut(test_content))
        print(jieba.cut(test_content, cut_all=True))
        print(jieba.cut(test_content, HMM=False))

        class MyCutter:
            def __init__(self):
                self.op: Callable = matx.text.Jieba()

            def __call__(self, a: str, cut_all: bool, HMM: bool) -> Any:
                return self.op(a, cut_all, HMM)

        op = matx.script(MyCutter)()
        print(op(test_content, False, True))
        print(op(test_content, True, False))
        print(op(test_content, False, False))

    def test_emoji_filter(self):
        class MyEmojiFilter:
            def __init__(self):
                user_emojis = ['[smile]']
                self.emoji: matx.text.EmojiFilter = matx.text.EmojiFilter(user_emojis=user_emojis)

            def __call__(self, s: str) -> Any:
                return self.emoji.filter(s)

        test_s = "hello[love], \U0001F1E6\U0001F1EB[smile]world"
        py_ret = MyEmojiFilter()(test_s)
        self.assertEqual(py_ret, "hello[love], world")
        tx_ret = matx.script(MyEmojiFilter)()(test_s)
        self.assertEqual(py_ret, tx_ret)

    def test_emoji_replace(self):
        class MyEmojiReplacer:
            def __init__(self):
                user_emojis = ['[smile]']
                self.emoji: matx.text.EmojiFilter = matx.text.EmojiFilter(user_emojis=user_emojis)

            def __call__(self, s: str, repl: str, keep_all: bool = False) -> Any:
                return self.emoji.replace(s, repl, keep_all)

        test_s = "hello[love], \U0001F1E6\U0001F1EB[smile]world"
        test_repl = '=='
        test_keep_all = False
        py_ret = MyEmojiReplacer()(test_s, test_repl, test_keep_all)
        self.assertEqual(py_ret, "hello[love], ==world")
        tx_ret = matx.script(MyEmojiReplacer)()(test_s, test_repl, test_keep_all)
        self.assertEqual(py_ret, tx_ret)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
