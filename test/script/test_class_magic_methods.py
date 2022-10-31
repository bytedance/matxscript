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
from typing import Dict, List


class TestClassMagicMethods(unittest.TestCase):

    def test_getitem(self):
        class MyList:
            def __init__(self) -> None:
                self.c: List[int] = [0, 1, 2]

            def __getitem__(self, idx: int) -> int:
                return self.c[idx]

            def __call__(self) -> int:
                return self.__getitem__(1)

        my_list_creator = matx.script(MyList)()
        assert my_list_creator() == 1


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
