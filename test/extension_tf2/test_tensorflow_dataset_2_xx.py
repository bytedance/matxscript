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
from typing import List, Tuple, Any


def parse_data_fn(x: bytes, i: int, f: float) -> matx.NDArray:
    print(x, i, f)
    return matx.NDArray([1] * 6, [2, 3], "int32")


def parse_data_fn_multi(x: bytes, i: int, f: float) -> Tuple[matx.NDArray, matx.NDArray]:
    print(x, i, f)
    a = matx.NDArray([1] * 6, [2, 3], "int32")
    b = matx.NDArray([1] * 6, [2, 3], "int32")
    return a, b


class TestTensorFlowModule2(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_tf_infer(self):
        tf_dataset_compiler = matx.extension.tensorflow.to_dataset_callback_op
        input_args = [b"hello", 1, 1.1]
        tx_op = matx.script(parse_data_fn)
        print(tx_op(*input_args))
        callback_fn = tf_dataset_compiler(tx_op, output_dtype=[tf.int32])
        tf_res = callback_fn(*input_args)
        print(tf_res)

    def test_tf_infer_multi(self):
        tf_dataset_compiler = matx.extension.tensorflow.to_dataset_callback_op
        input_args = [b"hello", 1, 1.1]
        tx_op = matx.script(parse_data_fn_multi)
        print(tx_op(*input_args))
        callback_fn = tf_dataset_compiler(tx_op, output_dtype=[tf.int32, tf.int32])
        tf_res = callback_fn(*input_args)
        print(tf_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
