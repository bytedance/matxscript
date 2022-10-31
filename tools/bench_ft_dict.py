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

from typing import Dict

import matx
from matx import FTDict


def bench_func_v1() -> object:
    r = 0.0
    cons: Dict[int, int] = {}
    for i in range(10000):
        cons[i] = i
    for i in range(100000):
        if i in cons:
            r += 1.0
    return r


def bench_func_v2() -> object:
    r = 0.0
    cons: FTDict[int, int] = {}
    for i in range(10000):
        cons[i] = i
    for i in range(100000):
        if i in cons:
            r += 1.0
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
