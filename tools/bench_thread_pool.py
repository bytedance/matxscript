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
from typing import List, Tuple
import time


class Op:
    def __init__(self) -> None:
        pass

    def process(self, item: str) -> bytes:
        for i in range(100):
            item.encode()
        return item.encode() + b'hi'

    def __call__(self, item: str) -> bytes:
        return self.process(item)


class BatchOp:
    def __init__(self) -> None:
        pass

    def process(self, item: str) -> bytes:
        for i in range(100):
            item.encode()
        return item.encode() + b'hi'

    def __call__(self, inputs: List) -> List:
        outputs = matx.List()
        outputs.reserve(len(inputs))
        for item in inputs:
            result = self.process(item)
            outputs.append(result)
        return outputs


class ParallelOp:
    __slots__: Tuple[matx.NativeObject, Op] = ['thread_pool', 'op']

    def __init__(self, pool_size: int, use_lockfree_pool: bool, op: Op) -> None:
        self.thread_pool = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)
        self.op = op

    def __call__(self, inputs: List) -> List:
        return self.thread_pool.ParallelFor(self.op, inputs)


def pipeline_multi_thread(inputs):
    op = matx.script(Op)()
    pool_size = 5
    use_lockfree_pool = True
    parallel_op = matx.script(ParallelOp)(pool_size, use_lockfree_pool, op)
    outputs = parallel_op(inputs)
    return outputs


def pipeline_single_thread(inputs):
    batch_op = matx.script(BatchOp)()
    outputs = batch_op(inputs)
    return outputs


def run_benchmark():
    warmup_times = 8
    inputs = [("toy_" + str(i)) for i in range(100)]
    test_case = {'inputs': inputs}

    jit_module_single_thread = matx.pipeline.Trace(pipeline_single_thread, inputs)
    output = jit_module_single_thread.run(test_case)
    for i in range(warmup_times):
        jit_module_single_thread.run(test_case)

    jit_module_multi_thread = matx.pipeline.Trace(pipeline_multi_thread, inputs)
    output = jit_module_multi_thread.run(test_case)
    for i in range(warmup_times):
        jit_module_multi_thread.run(test_case)

    jit_module_single_thread.profile(test_case)

    jit_module_single_thread.profile(test_case)


def trace(pipeline_func):
    import os
    script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    inputs = [("toy_" + str(i)) for i in range(10)]
    test_case = {'inputs': inputs}

    jit_module = matx.pipeline.Trace(pipeline_func, *[v for k, v in test_case.items()])
    jit_module.save(script_path + '/trace_output_lockfree', name='model.spec.json')

    output = jit_module.run(test_case)
    print(output)


if __name__ == '__main__':
    run_benchmark()

    # trace(pipeline_multi_thread)
    # trace(pipeline_single_thread)
