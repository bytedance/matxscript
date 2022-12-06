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
import random
import unittest
import matx


class Counter:
    """
    This class contains method that modifies a class member, and contains native __call__ method to test compatibility
    """

    def __init__(self, init_value: int) -> None:
        self.value: int = init_value

    def minus(self, a: int) -> int:
        self.value = self.value - a
        return self.value

    def add(self, a: int) -> int:
        self.value = self.value + a
        return self.value

    def __call__(self, a: int) -> int:
        self.value = self.value * a + a
        return self.value


class WorkFlow:
    def __init__(self, init_value=10, script=True):
        # create a foo instance
        if script:
            self.foo = matx.script(Counter)(init_value)
        else:
            self.foo = Counter(init_value=init_value)

    def process(self, a: int) -> int:
        # start to build computational graph
        # do some complicated things with c
        # d = do_something(c)
        d = self.foo(a)
        c = self.foo.minus(d)
        return self.foo.add(c)


class TestTracingClassMethod(unittest.TestCase):
    def test_tracing_class_method(self):
        init_value = 0

        workflow_matx = WorkFlow(init_value=init_value, script=True)
        workflow_process_mod = matx.trace(workflow_matx.process, 1)

        workflow_py = WorkFlow(init_value=init_value, script=False)
        workflow_py.process(1)  # to match up with trace side effects

        # generate random number and test
        test_steps = 100
        for _ in range(test_steps):
            random_int = random.randint(-10, 10)
            py_result = workflow_py.process(random_int)
            matx_result = workflow_process_mod(a=random_int)

            self.assertEqual(py_result, matx_result)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
