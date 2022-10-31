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

from typing import List
import matx
import time
import random
from matx import pipeline


@matx.script
def test_make_ngram(query: str, max_ngram_size: int) -> List:
    k = random.random()
    print("begin ", k, ": ", time.time())
    ngram_list = []
    positions = []
    for i in range(10000):
        query = query.strip()
        query_terms = query.split(' ')
        ngram_list = []
        positions = []

        for l in range(1, max_ngram_size + 1):
            for j in range(0, len(query_terms) - l + 1):
                ngram_list.append(" ".join(query_terms[j: j + l]))
                positions.append(i)
    print("end ", k, ": ", time.time())
    return ngram_list


def workflow(query):
    r1 = test_make_ngram(query, 2)
    r2 = test_make_ngram(r1[0], 2)
    r3 = test_make_ngram(query, 2)
    return r1, r2, r3


jit_mod = matx.pipeline.Trace(workflow, "hello world")
ret = jit_mod.warmup({"query": "hello world"})
ret = jit_mod.run({"query": "hello world"})
jit_mod.set_op_parallelism_threads(2)
ret = jit_mod.warmup({"query": "hello world"})
ret = jit_mod.run({"query": "hello world"})

# jit_mod.save("./my_pa_test")
# jit_mod = matx.pipeline.Load("./my_pa_test", -1)
# ret = jit_mod.run({"query": "hello world"})
