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
import numpy as np
import tensorflow as tf
import matx
from typing import List, Any

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class Adder(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64, name='x')])
    def add(self, x):
        return {"result": x + x}


class Generator(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64, name='x'),
                                  tf.TensorSpec(shape=[None], dtype=tf.int64, name='state')])
    def gen(self, x, state):
        # this is only an example, customize that according to your needs
        return {"result": x + state}


class MyGenerator:

    def __init__(self, tf_op: Any, state: matx.NDArray):
        self.tf_op: Any = tf_op
        self.state: matx.NDArray = state

    def __call__(self, x: matx.NDArray) -> List[matx.NDArray]:
        results = []
        for i in range(10):
            results.append(self.tf_op({"x": x, "state": self.state})["result"])
        return results


class TestTensorFlowModule2(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestTensorFlowModule2_%d/" % uuid.uuid4().int
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_tf_infer(self):
        model = Adder()
        save_path = self.work_path + "saved_model_v2"
        tf.saved_model.save(model, save_path)

        examples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        tf_outputs = model.add(examples)["result"]
        tf_outputs = tf_outputs.numpy()

        tf_op = matx.script(save_path, backend="TensorFlow", device=-1)
        tx_outputs = tf_op({"x": matx.array.from_numpy(examples)})["result"]
        tx_outputs = tx_outputs.numpy()

        self.assertTrue(np.alltrue(tf_outputs == tx_outputs))

    def test_script_tf_module(self):
        model = Adder()

        examples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        tf_outputs = model.add(examples)["result"]
        tf_outputs = tf_outputs.numpy()

        tf_op = matx.script(model, device=-1)
        tx_outputs = tf_op({"x": matx.array.from_numpy(examples)})["result"]
        tx_outputs = tx_outputs.numpy()

        self.assertTrue(np.alltrue(tf_outputs == tx_outputs))

    def test_tf_in_pipeline_control_flow(self):
        # build a tf model
        model = Generator()
        save_path = self.work_path + "generator_saved_model_v2"
        tf.saved_model.save(model, save_path)

        # make data examples
        state = np.array([1] * 10, dtype=np.int64)
        examples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        state = matx.array.from_numpy(state)
        examples = matx.array.from_numpy(examples)

        # make a matx-tf op
        tf_op = matx.script(save_path, backend="TensorFlow", device=-1)
        # compile a general python op
        generator = matx.script(MyGenerator)(tf_op=tf_op, state=state)

        # build our workflow
        def workflow(x):
            return generator(x)

        # just run workflow
        tx_outputs = workflow(examples)
        np_outputs = [x.numpy() for x in tx_outputs]
        print(np_outputs)

        # trace for server
        jit_mod = matx.trace(workflow, examples)
        print(jit_mod.run({"x": examples}))
        tx_save_path = self.work_path + "tf_control_flow_v2"
        jit_mod.save(tx_save_path)

        # load and test
        new_mod = matx.load(tx_save_path, device='cpu')
        print(new_mod.run({"x": examples}))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
