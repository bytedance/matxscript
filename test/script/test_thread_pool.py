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


def Op1(a: float) -> float:
    return 2 * a


class Op2:
    def __init__(self) -> None:
        pass

    def __call__(self, a: float) -> float:
        return 2 * a


class ParallelOp1:
    __slots__: Tuple[matx.NativeObject] = ['thread_pool']

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def __call__(self, op: Callable, inputs: List) -> List:
        return self.thread_pool.ParallelFor(op, inputs)


class ParallelOp2:
    __slots__: Tuple[matx.NativeObject] = ['thread_pool']

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def __call__(self, op: Op2, inputs: List) -> List:
        return self.thread_pool.ParallelFor(op, inputs)


class TestThreadPoolExecutor(unittest.TestCase):

    def test_parallel_for(self):
        def pipeline1(inputs):
            op_1 = matx.script(Op1)

            pool_size = 3
            use_lockfree_pool = True
            parallel_op = matx.script(ParallelOp1)(pool_size, use_lockfree_pool)

            outputs = parallel_op(op_1, inputs)
            return outputs

        def pipeline2(inputs):
            op_2 = matx.script(Op2)()

            pool_size = 3
            use_lockfree_pool = True
            parallel_op = matx.script(ParallelOp2)(pool_size, use_lockfree_pool)

            outputs = parallel_op(op_2, inputs)
            return outputs

        self.assertEqual(pipeline1([1.1, 2.1, 3.1]), matx.List([2.2, 4.2, 6.2]))
        self.assertEqual(pipeline2([1.1, 2.1, 3.1]), matx.List([2.2, 4.2, 6.2]))


class TestThreadPoolWithException(unittest.TestCase):

    def test_parallelfor_exception(self):

        def f(i: int) -> int:
            x = [1, 2, 3]
            return x[i]

        @matx.script
        def run_function() -> None:
            thread_pool = matx.make_native_object("ThreadPoolExecutor", 4, True)
            ret = thread_pool.ParallelFor(f, [0, 1, 2, 3])
            print(ret)

        try:
            run_function()
        except Exception as e:
            print(e)

    def test_submit(self):

        class MyTask:

            def __init__(self) -> None:
                pass

            def __call__(self) -> int:
                x = [1, 2]
                return x[2]

        @matx.script
        def run_submit() -> None:
            thread_pool = matx.make_native_object("ThreadPoolExecutor", 4, True)
            task = MyTask()
            f = thread_pool.Submit(task)
            print(f.get())

        try:
            run_submit()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
