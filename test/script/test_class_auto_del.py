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
from typing import Tuple, List


# python implemented resource, for sharing among python ops
class ToyResource:
    def __init__(self) -> None:
        pass

    def look(self, a: int) -> int:
        return a


# self-defined struct
class MyData:

    def __init__(self, query: str, score: float) -> None:
        self.query: str = query
        self.score: float = score


# op that use self-defined struct
class MyDataOp:
    def __init__(self) -> None:
        pass

    def __call__(self, query: str, score: float) -> List:
        result = []
        my_data = MyData(query, score)
        result.append(my_data)
        return result


def test():
    toy_resource = matx.script(ToyResource)()

    query = "hello"
    score = 0.3

    my_data_op = matx.script(MyDataOp)()
    a = my_data_op(query, score)


if __name__ == "__main__":
    test()
