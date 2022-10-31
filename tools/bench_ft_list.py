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
from matx import FTList


def minEditDistance_v1(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    cur: List[int] = [0] * (n + 1)
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


def minEditDistance_v2(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    cur: FTList[int] = [0] * (n + 1)
    # cur: FTList[int] = cur1 * (n + 1)
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


def bench_func_v1() -> object:
    a = "hello" * 2
    b = "world" * 2
    r = 0.0
    for i in range(100000):
        r += minEditDistance_v1(a, b)
    return r


def bench_func_v2() -> object:
    a = "hello" * 2
    b = "world" * 2
    r = 0.0
    for i in range(100000):
        r += minEditDistance_v2(a, b)
    return r


def get_exe_time(func_impl):
    # test case
    import time
    begin = time.time()
    func_impl()
    end = time.time()
    return "%.2f(ms)" % ((end - begin) * 1000)


def main():
    print("python vm1:", get_exe_time(bench_func_v1))
    print("matx v1:", get_exe_time(matx.script(bench_func_v1)))
    print("python vm2:", get_exe_time(bench_func_v2))
    print("matx v2:", get_exe_time(matx.script(bench_func_v2)))


if __name__ == "__main__":
    main()
