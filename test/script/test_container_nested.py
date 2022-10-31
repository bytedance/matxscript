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
from typing import Any


class TestContainerNested(unittest.TestCase):
    def test_construct(self):
        def construct() -> Any:
            l = [1, {2, 2, 4}, {3: 'a'}, (4, 5)]
            t = 2, 3
            return l, t

        def builtin_construct() -> Any:
            # tuple(iterable) is not supported now
            # l = list([1, set([2, 2, 4]), dict({3: 'a'}), tuple([4, 5])])
            # t = tuple({2, 3})
            l = list([1, set([2, 2, 4]), dict({3: 'a'}), (4, 5)])
            t = (2, 3)
            return l, t

        self.assertEqual(matx.script(construct)(), construct())
        self.assertEqual(matx.script(builtin_construct)(), builtin_construct())

    def test_argument(self):
        def as_argument() -> Any:
            l = []
            l.append({'qwe': 123})
            l[0]['zxc'] = 345
            l.append((4, 5))
            return l

        self.assertEqual(matx.script(as_argument)(), as_argument())

    def test_return(self):
        def as_return() -> list:
            return [1, {2, 2, 4}, {3: 'a'}, (4, 5)]

        self.assertEqual(matx.script(as_return)(), as_return())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
