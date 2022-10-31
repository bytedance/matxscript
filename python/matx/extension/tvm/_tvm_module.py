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
from .lib import compile_or_load_lib


class TVMModel(pipeline.ops.OpKernel):
    def __init__(self,
                 *,
                 location,
                 outputs):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TVMModel",
            location=location,
            outputs=outputs
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError("TVMModel is not a callable op")


class TVMInferOp(pipeline.ops.OpKernel):
    def __init__(self,
                 *,
                 models,
                 device,
                 batch_arg_name,
                 share_model):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TVMInferOp",
            models=models,
            device=device,
            batch_arg_name=batch_arg_name,
            share_model=share_model
        )

    def __call__(self, *args, **kwargs):
        return super(TVMInferOp, self).__call__(*args, **kwargs)


class TVMModule(object):
    def __init__(self,
                 device=None,
                 models=None,
                 batch_arg_name=None,
                 outputs=None,
                 share_model=True):
        super(TVMModule, self).__init__()
        assert isinstance(models, list)
        assert isinstance(batch_arg_name, str)
        assert isinstance(outputs, list)
        assert isinstance(device, int)

        compile_or_load_lib(silent=False)

        self.tvm_models = []
        self.device = device
        self.batch_arg_name = batch_arg_name
        self.outputs = outputs
        self.share_model = share_model
        self.model_holder = []
        for model_config in models:
            assert isinstance(model_config, dict)
            batch_size = model_config["batch_size"]
            model_path = model_config["model_path"]

            tvm_model = TVMModel(
                location=model_path, outputs=outputs)
            # keep tvm_model alive
            self.model_holder.append(tvm_model)
            self.tvm_models.append(
                {"batch_size": batch_size, "model_name": tvm_model.name})

    def make_pipeline_op(self):
        op = TVMInferOp(
            models=self.tvm_models,
            device=self.device,
            batch_arg_name=self.batch_arg_name,
            share_model=self.share_model
        )
        return op
