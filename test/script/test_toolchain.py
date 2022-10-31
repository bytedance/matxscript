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


from typing import Tuple
import unittest
import matx

from matx.contrib.inspect3_9_1_patch import getsource
from matx.toolchain import USE_SO_CACHE


class OP:
    def forward(self):
        return 'GlobalOP'


class ResourceManagerMatx:
    def __init__(self) -> None:
        pass


class Booster:
    def __init__(self) -> None:
        pass


class TextInfo:
    def __init__(self) -> None:
        pass


class FeatureExtractor:
    __slots__: Tuple[ResourceManagerMatx] = ['resource_mgr']

    def __init__(self, resource_mgr: ResourceManagerMatx) -> None:
        self.resource_mgr = resource_mgr


class GbdtRanker:
    __slots__: Tuple[object, object] = ['gbm', 'fe']

    def __init__(self, resource_mgr: ResourceManagerMatx) -> None:
        self.gbm = Booster()
        self.fe = FeatureExtractor(resource_mgr)
        pass

    def forward(self, text: TextInfo) -> None:
        pass


class TestScript(unittest.TestCase):

    def test_func(self):
        class OP:
            def forward(self):
                return 'InnerOP'

        op = OP()
        self.assertEqual(op.forward(), 'InnerOP')
        self.assertIn('InnerOP', getsource(OP))

        # This is the bug in python3.7 and dill.source
        # It prints the GlobalOP when getting source.
        import inspect
        print(inspect.getsource(OP))

    def test_codegen_dag(self):
        global USE_SO_CACHE
        USE_SO_CACHE = False

        for i in range(10):
            matx.script(GbdtRanker)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
