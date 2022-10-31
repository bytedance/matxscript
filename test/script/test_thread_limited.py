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
from typing import List, Tuple, Callable


def worker_func(a: float) -> float:
    return 2 * a


class WorkerOp:
    def __init__(self) -> None:
        pass

    def __call__(self, a: float) -> float:
        return 2 * a


class ParallelOp83:
    __slots__: Tuple[matx.NativeObject] = ['thread_pool']

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def __call__(self, op: Callable, inputs: List) -> List:
        return self.thread_pool.ParallelFor(op, inputs, 8, 3)


class ParallelOp93:
    __slots__: Tuple[matx.NativeObject] = ['thread_pool']

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def __call__(self, op: WorkerOp, inputs: List) -> List:
        return self.thread_pool.ParallelFor(op, inputs, 2, 3)


class TestThreadPoolExecutorLimited(unittest.TestCase):

    def test_parallel_for(self):
        def pipeline1(inputs):
            op_1 = matx.script(worker_func)

            pool_size = 3
            use_lockfree_pool = True
            parallel_op = matx.script(ParallelOp83)(pool_size, use_lockfree_pool)

            outputs = parallel_op(op_1, inputs)
            return outputs

        def pipeline2(inputs):
            op_2 = matx.script(WorkerOp)()

            pool_size = 3
            use_lockfree_pool = True
            parallel_op = matx.script(ParallelOp93)(pool_size, use_lockfree_pool)

            outputs = parallel_op(op_2, inputs)
            return outputs

        self.assertEqual(pipeline1([1.1, 2.1, 3.1] * 2), matx.List([2.2, 4.2, 6.2] * 2))
        self.assertEqual(pipeline2([1.1, 2.1, 3.1] * 2), matx.List([2.2, 4.2, 6.2] * 2))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
