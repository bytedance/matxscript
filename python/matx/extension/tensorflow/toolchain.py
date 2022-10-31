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

import sys

from matx import pipeline
from typing import Callable


def check(compiling_obj):
    tf = sys.modules.get('tensorflow')
    if tf is not None:
        if hasattr(tf, "Module") and isinstance(compiling_obj, (tf.Module,)):
            return True
        else:
            loaded_by_tf = False
            try:
                if isinstance(compiling_obj, tf.python.training.tracking.tracking.AutoTrackable):
                    loaded_by_tf = True
            except:
                return False
            if loaded_by_tf:
                raise TypeError(
                    "It seems that you are trying to script a model loaded by tf.saved_model, you should call matx.script(model_path, backend=\"TensorFlow\") instead.")
    return False


def script(compiling_obj, **kwargs):
    import tensorflow as tf
    from ._tensorflow_module import TensorFlowModule
    if isinstance(compiling_obj, str):
        mod = TensorFlowModule(location=compiling_obj, **kwargs)
    elif hasattr(tf, "Module") and isinstance(compiling_obj, (tf.Module,)):
        mod = TensorFlowModule(model=compiling_obj, **kwargs)
    else:
        raise NotImplementedError(type(compiling_obj))
    return mod.make_pipeline_op()


def _compile_or_load_lib_wrapper():
    from .lib import compile_or_load_lib
    compile_or_load_lib()


def get_dataset_op():
    from . import lib
    lib.compile_or_load_lib()
    return lib.get_dataset_op()


def to_dataset_callback_op(op, *, output_dtype=None) -> Callable:
    dataset_op_type = get_dataset_op()
    op_addr = op.native_op_handle
    op_addr_s = str(op_addr)

    class Fn:
        def __init__(self):
            self.op = op

        def __call__(self, *args):
            return dataset_op_type(op_addr=op_addr_s,
                                   input_args=args,
                                   output_dtype=output_dtype)

    return Fn()


pipeline.PluginLoader.register("TFInferOp", _compile_or_load_lib_wrapper)
pipeline.PluginLoader.register("TFModel", _compile_or_load_lib_wrapper)
