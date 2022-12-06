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
import uuid
import matx
from matx import pipeline
from typing import List, Dict

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestCommonTrace(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestCommonTrace_%d/" % uuid.uuid4().int

    def test_session_attr(self):

        def workflow(a, b, **kwargs):
            dd = kwargs["dd"]
            return a, b, dd

        jit_module = pipeline.Trace(workflow, 1, 2, dd=1)
        jit_module.set_sess_attr("m3_retty", 12)  # 12 means kBatchTensorMap
        jit_module.set_sess_attr("model_name", "tiny")
        jit_module.set_sess_attr("allowed_batch_size", matx.List([4, 8, 16]))

        ret = jit_module.run({"a": 1, "b": 2, "dd": 10})
        assert ret == (1, 2, 10)

        save_path = self.work_path + "/test_session_attr"
        jit_module.save(save_path)
        new_jit_mod = matx.pipeline.Load(save_path, -1)

        assert new_jit_mod.get_sess_attr("m3_retty") == 12
        assert new_jit_mod.get_sess_attr("model_name") == "tiny"
        assert new_jit_mod.has_sess_attr("m3_retty")
        assert not new_jit_mod.has_sess_attr("m4_retty")
        assert len(new_jit_mod.get_sess_attr("allowed_batch_size")) == 3

    def test_trace_container_constructor(self):

        @matx.script
        def sub(a: int, b: int) -> int:
            return a - b

        @matx.script
        def add(a: int, b: int) -> int:
            return a + b

        @matx.script
        def sum(l: List) -> int:
            s = 0

            s += l[0]
            s += l[1]
            for x in l[2]:
                s += x
            s += l[3][0]
            s += l[3][1]
            for x in l[3][2]:
                s += x
            for kv in l[3][3].items():
                k, v = kv
                s += k
                s += v

            return s

        def process(a: int, b: int):
            m = sub(a, b)
            n = add(m, m)
            k = sum([m, n, [1, 2], [1, n, {m, n, 10}, {m: 10, 100: n}]])

            return k

        save_path = self.work_path + "/test_trace_container_constructor"
        m1 = matx.trace(process, 20, 10)
        self.assertEqual(m1.run({"a": 20, "b": 10}), 224)
        m1.save(save_path)
        m2 = matx.load(save_path, "cpu")
        self.assertEqual(m2.run({"a": 20, "b": 10}), 224)

    def test_trace_container_getitem_by_constant(self):
        @matx.script
        def sum(l: List) -> int:
            s = 0
            for x in l:
                s += x

            return s

        @matx.script
        def add(a: int, b: int) -> int:
            return a + b

        @matx.script
        def duplicate(x: int) -> List:
            return [x, x, x, x, x]

        def process(a: int):
            l = duplicate(a)
            m = sum(l[:])
            n = sum(l[::2])
            j = add(m, n)
            k = add(l[0], l[1])
            return add(j, k)

        save_path = self.work_path + "/test_trace_container_getitem"
        m = pipeline.Trace(process, 42)
        self.assertEqual(m.run({'a': 42}), 420)
        m.save(save_path)
        m = pipeline.Load(save_path, -1)
        self.assertEqual(m.run({'a': 42}), 420)

    def test_trace_container_getitem_by_var(self):
        @matx.script
        def make_list(x: str, y: str, z: str) -> List:
            return [x, y, z]

        @matx.script
        def make_dict(x: str, y: str, z: str) -> Dict:
            return {x: x, y: y, z: z}

        @matx.script
        def concat(l: List) -> str:
            s = ''
            for x in l:
                s += x
            return s

        def process(x: str, y: str, z: str):
            l = make_list(x, y, z)
            d = make_dict(x, y, z)
            m = concat(l)
            n = concat(l[::2])
            j = concat([m, n])
            k = concat([d[x], d[y]])
            return concat([j, k])

        save_path = self.work_path + "/test_trace_container_getitem_by_var"
        m = pipeline.Trace(process, 'a', 'b', 'z')
        self.assertEqual(m.run({'x': 'a', 'y': 'b', 'z': 'c'}), 'abcacab')
        m.save(save_path)
        m = pipeline.Load(save_path, -1)
        self.assertEqual(m.run({'x': 'a', 'y': 'b', 'z': 'c'}), 'abcacab')

    def test_trace_container_getitem_nested(self):
        @matx.script
        def make_ndarray(dtype: str) -> matx.NDArray:
            return matx.NDArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], dtype)

        def process(dtype: str, index: int):
            nd = make_ndarray(dtype)
            return nd[index][index][index]

        save_path = self.work_path + "/test_trace_container_getitem_nested"
        m = pipeline.Trace(process, 'int32', 0)
        self.assertEqual(m.run({'dtype': 'int32', 'index': 0}), 1)
        m.save(save_path)
        m = pipeline.Load(save_path, -1)
        self.assertEqual(m.run({'dtype': 'int32', 'index': 1}), 8)

    def test_make_session_by_one_function(self):

        def add(x: int, y: int) -> int:
            return x + y

        sess = matx.toolchain.make_session(add)
        assert sess.run({'x': 2, 'y': 3}) == 5
        assert sess.input_names == ['x', 'y']

    def test_make_session_by_one_class(self):

        class AddX:
            def __init__(self, x: int) -> None:
                self._x: int = x

            def __call__(self, x: int) -> int:
                return self._x + x

        sess = matx.toolchain.make_session(AddX)(7)
        assert sess.run({'x': 2}) == 9
        assert sess.input_names == ['x']

    def test_make_session_by_one_class_method(self):

        class AddX:
            def __init__(self, x: int) -> None:
                self._x: int = x

            def add(self, x: int) -> int:
                return self._x + x

        sess = matx.toolchain.make_session(AddX, method='add')(7)
        assert sess.run({'x': 2}) == 9
        assert sess.input_names == ['x']


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
