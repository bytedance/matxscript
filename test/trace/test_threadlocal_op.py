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

import os
import unittest
import time
import matx
from matx import pipeline

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestThreadLocalOp(unittest.TestCase):

    def test_load_share_op(self):
        model_path = SCRIPT_PATH + "/../tempdir/TestThreadLocalOp/share"

        class MyShareOp:

            def __init__(self) -> None:
                self.create_time: int = int(time.time() * 1000000000)
                print("call MyShareOp.__init__", self.create_time)

            def __call__(self) -> int:
                return self.create_time

        mod_op = matx.script(MyShareOp, share=True)()
        print(mod_op())

        def process():
            return mod_op()

        jit_mod = pipeline.Trace(process)
        print(jit_mod.run({}))
        jit_mod.save(model_path)

        jit_mod1 = pipeline.Load(model_path, -1)
        time.sleep(0.1)
        jit_mod2 = pipeline.Load(model_path, -1)
        ret1 = jit_mod1.run({})
        ret2 = jit_mod2.run({})
        time.sleep(0.1)
        self.assertEqual(ret1, ret2)

    def test_load_thread_local_op(self):
        model_path = SCRIPT_PATH + "/../tempdir/TestThreadLocalOp/local"

        class MyThreadLocalOp:

            def __init__(self) -> None:
                self.create_time: int = int(time.time() * 1000000000)
                print("call MyThreadLocalOp.__init__", self.create_time)

            def __call__(self) -> int:
                return self.create_time

        mod_op = matx.script(MyThreadLocalOp, share=False)()
        print(mod_op())

        def process():
            return mod_op()

        jit_mod = pipeline.Trace(process)
        print(jit_mod.run({}))
        jit_mod.save(model_path)

        jit_mod1 = pipeline.Load(model_path, -1)
        time.sleep(0.1)
        jit_mod2 = pipeline.Load(model_path, -1)
        ret1 = jit_mod1.run({})
        ret2 = jit_mod2.run({})
        time.sleep(0.1)
        self.assertNotEqual(ret1, ret2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
