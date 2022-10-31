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

from matx import pipeline


def check(compiling_obj):
    return False


def script(*args, **kwargs):
    from ._tvm_module import TVMModule
    mod = TVMModule(*args, **kwargs)
    return mod.make_pipeline_op()


def _compile_or_load_lib_wrapper():
    from .lib import compile_or_load_lib
    compile_or_load_lib()


pipeline.PluginLoader.register("TVMInferOp", _compile_or_load_lib_wrapper)
pipeline.PluginLoader.register("TVMModel", _compile_or_load_lib_wrapper)
