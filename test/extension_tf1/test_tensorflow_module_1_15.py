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

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestTensorFlowModule1(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestTensorFlowModule1_%d/" % uuid.uuid4().int
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_tf_infer(self):
        """
        weight = [2, 0, 0, 2]
        def f(x, y)
           a = weight * x
           b = - weight * y
           return a, b
        """
        save_path = self.work_path + "saved_model_v1"
        np_x = np.random.rand(2, 2).astype(dtype=np.float32)
        np_y = np.random.rand(2, 2).astype(dtype=np.float32)

        weight = tf.get_variable(
            'weight', shape=[2, 2],
            initializer=tf.constant_initializer([2, 0, 0, 2]))
        x = tf.placeholder(tf.float32, shape=(2, 2), name='x')
        y = tf.placeholder(tf.float32, shape=(2, 2), name='y')
        a = tf.matmul(weight, x, name='a')
        b = tf.negative(tf.matmul(weight, y), name='b')
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run([a, b], feed_dict={x: np_x, y: np_y})
            tf.saved_model.simple_save(
                sess, save_path,
                inputs={'x': x, 'y': y},
                outputs={'a': a, 'b': b}
            )
        tf_op = matx.script(save_path, backend="TensorFlow", device=-1)
        print(tf_op({"x": matx.array.from_numpy(np_x), "y": matx.array.from_numpy(np_y)}))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
