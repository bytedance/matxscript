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
import sys
from matx import pipeline

from .lib import compile_or_load_lib


class TFModel(pipeline.ops.OpKernel):
    """Create TFModel Object

    Parameters
    ----------

    location : str
        The TF SavedModel path

    use_xla : int
        whether enable tf xla

    allow_growth : bool
        Allow GPU memory to grow on demand

    """

    def __init__(self,
                 *,
                 location=None,
                 use_xla=1,
                 allow_growth=True):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TFModel",
            location=location,
            use_xla=use_xla,
            allow_growth=allow_growth
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError("TFModel is not a callable op")


class TFInferOp(pipeline.ops.OpKernel):
    """Create TFInferOp

    Parameters
    ----------
    model : str
        TFModel name

    device : int
        GPU serial numbers, or -1(CPU)

    """

    def __init__(self,
                 *,
                 model,
                 device=None):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TFInferOp",
            model=model,
            device=device
        )

    def __call__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args : Tuple(Dict[str, Tensor], ...)
            input TensorMaps

        kwargs : Optional
            Not supported currently

        Returns
        -------
        result : Dict[str, Tensor]
            the result TensorMap

        """
        return super(TFInferOp, self).__call__(*args, **kwargs)


class TensorFlowModule(object):
    """TensorFlowModule

    Parameters
    ----------
    device : int
        GPU serial numbers, or -1(CPU)

    model : tf.Module
        The TF Module instance

    location : str
        The TF SavedModel path

    use_xla : int
        whether enable tf xla

    allow_growth : bool
        Allow GPU memory to grow on demand

    """

    def __init__(self,
                 *,
                 device=None,
                 model=None,
                 location=None,
                 use_xla=1,
                 allow_growth=True):
        assert model is None or location is None, "model and location should not be set at the same time"
        super(TensorFlowModule, self).__init__()
        # init tf
        compile_or_load_lib(silent=False)
        if model is not None:
            import time
            import shutil
            import atexit
            import tensorflow as tf
            timestamp = int(round(time.time() * 1000))
            location = f"./tf_module_{timestamp}"
            tf.saved_model.save(model, location)
            atexit.register(shutil.rmtree, location)

        self._tf_model = TFModel(location=location,
                                 use_xla=use_xla,
                                 allow_growth=allow_growth)
        self._device = device
        self.__holder = sys.modules['matx']

    def make_pipeline_op(self):
        op = TFInferOp(
            model=self._tf_model.name,
            device=self._device
        )
        return op
