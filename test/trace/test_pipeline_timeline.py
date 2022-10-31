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
import matx
from matx import pipeline
from typing import List


def test_make_ngram(query: str, max_ngram_size: int) -> List:
    ngram_list = matx.List()
    positions = matx.List()
    for i in range(10000):
        query = query.strip()
        query_terms = query.split(' ')
        ngram_list = matx.List()
        positions = matx.List()

        for l in range(1, max_ngram_size + 1):
            for j in range(0, len(query_terms) - l + 1):
                ngram_list.append(" ".join(query_terms[j: j + l]))
                positions.append(i)

    return ngram_list


class TestPipelineTimeline(unittest.TestCase):

    def test_pipeline_timeline(self):
        def my_pipeline(query: str):
            op1 = matx.script(test_make_ngram)
            op2 = matx.script(test_make_ngram)
            n6_list = op1(query, 6)
            n4_list = op2(query, 4)
            return n6_list, n4_list

        jit_mod = pipeline.Trace(my_pipeline, "hello world")
        meta = jit_mod.gen_step_meta({"query": "hello world"})
        jit_mod.gen_timeline(meta)

        jit_mod.profile({"query": "hello world"})


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
