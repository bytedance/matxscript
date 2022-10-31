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

const_none = None
const_int = 0
const_float = 0.1
const_bool = True
const_str = "hello"
const_list = [1, 2, 3]

const_dict = {"h": "h"}
const_tuple = (1,)


def add(a: float, b: float) -> float:
    return a + b


class MyMod(object):

    def __init__(self):
        super(MyMod, self).__init__()
        self.a: int = 0

    def foo(self) -> int:
        return self.a
