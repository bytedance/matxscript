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
import numpy as np
import pickle


class TestPickle(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_pickle_list(self):
        a1 = matx.List([1] * 100)
        pickle_data = pickle.dumps(a1)
        a2 = pickle.loads(pickle_data)
        self.assertEqual(a1, a2)

    def test_pickle_dict(self):
        a1 = matx.Dict({"hello": "world"})
        pickle_data = pickle.dumps(a1)
        a2 = pickle.loads(pickle_data)
        self.assertEqual(a1, a2)

    def test_pickle_set(self):
        a1 = matx.Set([1, 2, 3])
        pickle_data = pickle.dumps(a1)
        a2 = pickle.loads(pickle_data)
        self.assertEqual(a1, a2)

    def test_pickle_ndarray(self):
        a1 = matx.NDArray(1, shape=[2, 3], dtype="int32")
        pickle_data = pickle.dumps(a1)
        a2 = pickle.loads(pickle_data)
        self.assertTrue(np.alltrue(a1.numpy() == a2.numpy()))

    def test_pickle_nested(self):
        py_data = [None, 1, 1.1, 'hello', b'hi', [0, "hello"], {"h": 1}]
        tx_data = matx.to_runtime_object(py_data)
        pickle_data = pickle.dumps(tx_data)
        new_data = pickle.loads(pickle_data)
        self.assertEqual(tx_data, new_data)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
