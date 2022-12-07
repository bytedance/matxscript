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
from typing import Dict, List

import matx


class Text2Ids:
    def __init__(self) -> None:
        self.table: Dict[str, int] = {
            "hello": 0,
            "world": 1,
            "[UNK]": 2,
        }

    def lookup(self, word: str) -> int:
        return self.table.get(word, 2)

    def batch_lookup(self, words: List[str]) -> List[int]:
        return [self.lookup(w) for w in words]


class WorkFlow:
    def __init__(self, script=True):
        # compile
        if script:
            self.text2ids = matx.script(Text2Ids)()
        else:
            self.text2ids = Text2Ids()

    def process(self, texts):
        ids = self.text2ids.batch_lookup(texts)
        return ids


class TestText2Id(unittest.TestCase):
    def testText2Id(self):
        examples = "hello world unknown".split()
        # test
        handler_py = WorkFlow(script=False)
        handler_matx = WorkFlow(script=True)
        mod = matx.trace(handler_matx.process, examples)

        self.assertListEqual(handler_py.process(examples), list(mod(texts=examples)))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
