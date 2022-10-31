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
from matx import pipeline


class MySimplePipelineOpTest:
    __slots__ = []

    def __init__(self) -> None:
        pass

    def __call__(self, a: int) -> int:
        return a * 2


class TestOpReuseInPipeline(unittest.TestCase):

    def test_reuse_pipeline_op(self):
        op = matx.script(MySimplePipelineOpTest)()

        def process(a: int):
            b = op(a)
            c = op(a)
            return b, c

        print(matx.pipeline.Trace(process, 10))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
