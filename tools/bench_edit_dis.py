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
import matx
import timeit
from matx import FTList
import torch


def edit_distance_raw(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)

    cur = [0] * (n + 1)

    for k in range(1, n + 1):
        cur[k] = k
    for i in range(1, m + 1):
        pre = cur[0]
        cur[0] = i
        for j in range(1, n + 1):
            temp = cur[j]
            if word1[i - 1] == word2[j - 1]:
                cur[j] = pre
            else:
                cur[j] = min(pre, cur[j - 1], cur[j]) + 1
            pre = temp
    return cur[n]


def edit_distance_ft(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)

    cur: FTList[int] = [0] * (n + 1)

    for k in range(1, n + 1):
        cur[k] = k
    for i in range(1, m + 1):
        pre = cur[0]
        cur[0] = i
        for j in range(1, n + 1):
            temp = cur[j]
            if word1[i - 1] == word2[j - 1]:
                cur[j] = pre
            else:
                cur[j] = min(pre, cur[j - 1], cur[j]) + 1
            pre = temp
    return cur[n]


edit_distance_tx = matx.script(edit_distance_raw)
edit_distance_tx_ft = matx.script(edit_distance_ft)
edit_distance_torch = torch.jit.script(edit_distance_raw)


def bench_entry(func_impl, prefix):
    for i in range(100):
        func_impl("hello, world", "hi, world")
    print(prefix, " : ", timeit.timeit(lambda: func_impl("hello, world", "hi, world"), number=1000))


bench_entry(edit_distance_raw, "Python")
bench_entry(edit_distance_tx, "MATX")
bench_entry(edit_distance_tx_ft, "MATX FT")
bench_entry(edit_distance_torch, "PyTorch")
