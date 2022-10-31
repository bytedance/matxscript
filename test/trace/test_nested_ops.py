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
from typing import Tuple


class MyInfo:
    def __init__(self) -> None:
        pass

    def process(self) -> int:
        return 2

    def __call__(self) -> int:
        return 2


class MyOp:
    __slots__: Tuple[MyInfo, MyInfo] = ['my_info1', 'my_info2']

    def __init__(self, my_info2: MyInfo) -> None:
        self.my_info1 = MyInfo()
        self.my_info2 = my_info2

    def __call__(self, my_info: MyInfo, a: int) -> int:
        return self.my_info1.process() + self.my_info2.process() + my_info.process() + a


def pipeline():
    my_info = matx.script(MyInfo)()
    my_info.process()
    my_op = matx.script(MyOp)(my_info)

    result = my_op(my_info, 1)
    print(result)


if __name__ == "__main__":
    pipeline()
