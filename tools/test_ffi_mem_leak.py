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

import time
import matx


def test_mem_leak(arg0: object,
                  arg1: object,
                  arg2: object,
                  arg3: object,
                  arg4: object,
                  arg5: object,
                  arg6: object,
                  arg7: object,
                  arg8: object,
                  arg9: object) -> object:
    return arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9


def test_func():
    arg0 = b"this is a test"
    arg1 = "this is a test"
    arg2 = ("this is a test", b"hello", 1, 2.5)
    arg2_obj = matx.to_runtime_object(arg2)
    arg3 = ["this is a test", b"hello", 1, 2.5]
    arg3_obj = matx.to_runtime_object(arg2)
    arg4 = {"hello": 21, b"tt": arg2, 45: ["hello", arg2]}
    arg4_obj = matx.to_runtime_object(arg3)
    i = 0
    test_native_func = matx.script(test_mem_leak)
    while True:
        test_native_func(arg0, arg1, arg2, arg3, arg4, None, 0, 1.0, True, False)
        test_native_func(arg0, arg1, arg2_obj, arg3_obj, arg4_obj, None, 0, 1.0, True, False)
        i += 1
        if i % 100000 == 0:
            print(f"loops: {i}")
            time.sleep(1)


if __name__ == "__main__":
    test_func()
