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


def test_simple_add(a, b, c):
    return a + b + c


def test_container(c):
    d = matx.Dict({7: 9})
    a = matx.List()
    a.append(b"22")
    a.append(1)
    a.append(0.1)
    b = matx.List()
    b.append(a)
    b.append(d)
    return b, c


class MyData:

    def __init__(self):
        self.a = 1

    def __call__(self, *args):
        c = 0
        for a in args:
            c += a
        return self.a + c + 10

    def __del__(self):
        print("__del__", id(self))


def test_main():
    # test simple function
    py_ret = test_simple_add(1, 2, 3.0)
    tx_func = matx.to_packed_func(test_simple_add)
    tx_ret = tx_func(1, 2, 3.0)
    assert py_ret == tx_ret

    # test simple function with container
    py_ret = test_container([1, 2, 3])
    tx_func = matx.to_packed_func(test_container)
    tx_ret = tx_func([1, 2, 3])
    assert py_ret == tx_ret

    # test callable object
    d = MyData()
    py_ret = d(1, 2, 3)
    tx_func = matx.to_packed_func(d)
    tx_ret = tx_func(1, 2, 3)
    assert py_ret == tx_ret


if __name__ == "__main__":
    test_main()
