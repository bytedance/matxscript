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
from typing import Any


class MyTokenizerTask:

    def __init__(self, query: str, titles: List[str], sep: str = " ") -> None:
        self.query: str = query
        self.titles: List[str] = titles
        self.sep: str = sep

    def __call__(self, i: int = 0) -> Tuple[int, List[str], List[List[str]]]:
        qs = self.query.split(self.sep)
        ts = []
        for t in self.titles:
            ts.append(t.split(self.sep))
        return i, qs, ts


class MyTaskManager:

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool: matx.NativeObject = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def submit(self, task: MyTokenizerTask) -> matx.NativeObject:
        return self.thread_pool.Submit(task, 1)


class MyTokenizerOp:

    def __init__(self) -> None:
        self.task_manager: MyTaskManager = MyTaskManager(2, True)

    def __call__(self, queries: List[str], titles: List[List[str]]) -> Any:
        assert len(queries) == len(titles), "q/t mismatch"
        futures = []
        for i in range(len(queries)):
            task = MyTokenizerTask(queries[i], titles[i])
            futures.append(self.task_manager.submit(task))

        result = []
        for f in futures:
            result.append(f.get())
        return result


class TestThreadPoolAsync(unittest.TestCase):

    def test_submit_task(self):
        def my_pipeline(queries, titles):
            op_1 = matx.script(MyTokenizerOp)()
            return op_1(queries, titles)

        q_l = ["hello world", "hi word"]
        t_l = [["11 22"], ["kk jj"]]
        print(my_pipeline(q_l, t_l))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
