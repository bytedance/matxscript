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
import atexit
import logging
import numpy as np
import matx
from matx import pipeline
from matx.pipeline import _ffi_api, make_op_creator_function
from matx.pipeline.symbol import Symbol, Variable, Constant, BaseSymbol, make_symbol
from matx.pipeline._tracing_state import tracing

from .lib import compile_or_load_lib

_cc_op_creator = make_op_creator_function("PythonBaseOp")


class PyTorchDeviceNotFoundError(ValueError):
    """Raised when the device could not be acquired from a torch model."""

    def __init__(self, model) -> None:
        self.model = model

    def __str__(self) -> str:
        return f"The device could not be acquired from torch model {self.model}, please pass the device parameter."


class PyTorchDeviceNotSetError(ValueError):
    """Raised when the device parameter is not passed by user."""

    def __init__(self, model) -> None:
        self.model = model

    def __str__(self) -> str:
        return f"Loss the device parameter, model: '{self.model}'"


class PyTorchRuntimeError(RuntimeError):
    """Raised when the device parameter is not passed by user."""

    def __init__(self, model, stack_message) -> None:
        self.model = model
        self.stack_message = stack_message

    def __str__(self) -> str:
        errmsg = self.stack_message[self.stack_message.find('\n') + 1:]
        return f"{errmsg}"


class TorchModel(pipeline.ops.OpKernel):
    """Create TorchModel

    Parameters
    ----------
    location : str
        PyTorch jit model path

    """

    def __init__(self,
                 *,
                 location,
                 example):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TorchModel",
            location=location,
            example=example,
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError("TorchModel is not a Callable Op")


class TorchInferOp(pipeline.ops.OpKernel):
    """Create TorchInferOp

    Parameters
    ----------
    model : TorchModel
    device: int, str
    output_to_cpu: bool
    """

    def __init__(self,
                 *,
                 model=None,
                 device=None,
                 output_to_cpu=True):
        compile_or_load_lib(silent=False)
        super().__init__(
            "TorchInferOp",
            model=model,
            device=device,
            output_to_cpu=output_to_cpu,
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
        return super(TorchInferOp, self).__call__(*args, **kwargs)


class PyTorchInferOp(pipeline.ops.OpKernel):
    """Create PyTorchInferOp

    Parameters
    ----------
    impl : pipeline.ops.OpKernel
        Implement op

    """

    def __init__(self,
                 *,
                 impl):
        compile_or_load_lib(silent=False)
        super().__init__(
            "PyTorchInferOp",
            impl=impl,
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
        return super(PyTorchInferOp, self).__call__(*args, **kwargs)


def make_pipeline_op_from_location(location=None,
                                   device=None,
                                   output_to_cpu=True,
                                   **kwargs):
    mod = TorchModel(location=location, example=None)
    op = TorchInferOp(model=mod.name, device=device, output_to_cpu=output_to_cpu)
    return PyTorchInferOp(impl=op)


class PytorchModule(object):
    """
    """

    def __init__(
            self,
            *,
            model=None,
            trace_func=None,
            location=None,
            device=None,
            output_to_cpu=True):
        compile_or_load_lib(silent=False)
        super().__init__()
        self._already_traced = False
        import time
        timestamp = int(round(time.time() * 1000))
        # init torch
        import torch
        self._torch = torch
        self._torch_from_numpy = torch.from_numpy
        self._torch_Tensor = torch.Tensor
        self._torch_jit_trace = torch.jit.trace if not trace_func else trace_func
        self._need_trace = False
        self._need_save_tmp = False
        self._output_to_cpu = output_to_cpu

        # init model device
        if device is None:
            if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
                try:
                    self._torch_device = next(model.parameters()).device
                except StopIteration:
                    raise PyTorchDeviceNotFoundError(model)
                if self._torch_device.type == "cuda":
                    device = self._torch_device.index
                else:
                    device = -1
            else:
                raise PyTorchDeviceNotSetError("device not set")
        else:
            if isinstance(device, int):
                if device >= 0:
                    self._torch_device = torch.device('cuda:%d' % device)
                else:
                    self._torch_device = torch.device('cpu')
            elif isinstance(device, str):
                self._torch_device = torch.device(device)
            else:
                raise TypeError(f"invalid device: {device}")

        # init model
        if model is None:
            if not os.path.exists(location):
                raise ValueError(
                    "model is None and not found location: %s" % location)
            with torch.no_grad():
                self._model = torch.jit.load(location, self._torch_device)
        else:
            self._model = model
            if isinstance(model, torch.jit.ScriptModule):
                self._need_trace = False
                self._need_save_tmp = True
                self._set_device()
            elif isinstance(model, torch.nn.Module):
                self._need_trace = True
                self._need_save_tmp = True
                self._set_device()
            else:
                self._need_trace = True
                self._need_save_tmp = True
                logging.warning(
                    "can't set device for non torch.nn.Module, make sure the device of input model equals input device")

        # init matx cpp resource name and location
        if location is None:
            location = f"./torch_jit_{timestamp}.jit"

        # init matx cpp pass op and resource
        self._py_r_name = "BaseTorchModel"
        self._pass_r_name = "TorchModel"
        self._pass_r_options = dict(
            location=location,
        )
        self._py_base_r = _cc_op_creator(py_op_name=self._py_r_name,
                                         pass_op_name=self._pass_r_name,
                                         pass_op_options=self._pass_r_options)

        self._py_op_name = "BaseTorchInferOp"
        self._pass_op_name = "TorchInferOp"
        self._pass_op_options = dict(
            model=self._py_base_r.name,
            device=device,
            output_to_cpu=output_to_cpu
        )

    def _set_device(self):
        model = self._model.to(self._torch_device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self._model = model

    def _trans_list_(self, data):
        new_data = list()
        for v in data:
            new_data.append(self._trans_(v))
        return new_data

    def _trans_dict_(self, data):
        new_data = dict()
        for k, v in data.items():
            new_data[k] = self._trans_(v)
        return new_data

    def _trans_set_(self, data):
        new_data = set()
        for v in data:
            new_data.add(v)
        return new_data

    def _trans_tuple_(self, data):
        new_data = list()
        for v in data:
            new_data.append(self._trans_(v))
        return tuple(new_data)

    def _trans_(self, data):
        if isinstance(data, BaseSymbol):
            return self._trans_(data.data_2_71828182846)
        elif isinstance(data, (tuple, matx.Tuple)):
            raise NotImplementedError("Tuple is not supported as model input")
        elif isinstance(data, (list, matx.List)):
            return self._trans_list_(data)
        elif isinstance(data, (dict, matx.Dict)):
            return self._trans_dict_(data)
        elif isinstance(data, (set, matx.Set)):
            return self._trans_set_(data)
        else:
            if isinstance(data, matx.array.NDArray):
                return data.torch(copy=False).to(self._torch_device)
            elif isinstance(data, np.ndarray):
                return self._torch_from_numpy(data).to(self._torch_device)
            else:
                return data

    def _reverse_trans_list_(self, data):
        new_data = matx.List()
        for v in data:
            new_data.append(self._reverse_trans_(v))
        return new_data

    def _reverse_trans_dict_(self, data):
        new_data = matx.Dict()
        for k, v in data.items():
            new_data[k] = self._reverse_trans_(v)
        return new_data

    def _reverse_trans_tuple_(self, data):
        new_data = list()
        for v in data:
            new_data.append(self._reverse_trans_(v))
        return tuple(*new_data)

    def _reverse_trans_(self, data):
        if isinstance(data, tuple):
            raise NotImplementedError(
                "Tuple is only supported as first level output")
        if isinstance(data, list):
            return self._reverse_trans_list_(data)
        elif isinstance(data, dict):
            return self._reverse_trans_dict_(data)
        else:
            if isinstance(data, self._torch_Tensor):
                import torch.utils.dlpack
                if data.dim() == 0:
                    # 0-dim tensor is not supported
                    data = data.unsqueeze(-1)
                if self._output_to_cpu:
                    return matx.array.from_dlpack(
                        torch.utils.dlpack.to_dlpack(data.detach().cpu()))
                else:
                    return matx.array.from_dlpack(
                        torch.utils.dlpack.to_dlpack(data.detach()))
            else:
                return data

    def _reverse_trans_output_(self, model_output):
        if isinstance(model_output, tuple):
            assert len(model_output) > 0
            tmp = []
            for item in model_output:
                tmp.append(self._reverse_trans_(item))
            return tuple(tmp)
        return self._reverse_trans_(model_output)

    def __call__(self, *args):
        with self._torch.no_grad():
            args_data = []
            for arg in args:
                args_data.append(self._trans_(arg))
            try:
                model_output = self._model(*args_data)
            except:
                import traceback
                msg = traceback.format_exc()
                raise PyTorchRuntimeError(self._model, msg)
            res = self._reverse_trans_output_(model_output)
            if self._already_traced or not tracing():
                return res
            self._already_traced = True
            if "example" not in self._pass_r_options:
                model_args = [self._reverse_trans_(_arg) for _arg in args_data]
                example = matx.to_runtime_object(tuple(model_args))
                self._pass_r_options["example"] = example
                _ffi_api.PythonBaseOp_UpdatePassOpOptions(
                    self._py_base_r.native_op,
                    self._pass_r_options,
                )
            if self._need_trace:
                model_jit = self.Trace(args_data)
            else:
                model_jit = self._model

            if self._need_save_tmp:
                location = self._pass_r_options["location"]
                p = os.path.dirname(location)
                if not os.path.exists(p):
                    os.makedirs(p)
                print("begin save tmp model jit: ", location)
                model_jit.save(location)
                atexit.register(os.remove, location)
                print("finish save tmp model jit: ", location)
        return res

    def Trace(self, inputs):
        """internal use

        Returns
        -------

        """
        model_jit = None
        import torch
        if self._torch_jit_trace == torch.jit.script:
            with torch.no_grad():
                model_jit = torch.jit.script(self._model)
        else:
            with torch.no_grad():
                model_jit = self._torch_jit_trace(self._model, inputs)
        return model_jit

    def make_pipeline_op(self):
        py_callable = matx.to_packed_func(self)
        py_base_op = _cc_op_creator(py_callable=py_callable,
                                    py_op_name=self._py_op_name,
                                    pass_op_name=self._pass_op_name,
                                    pass_op_options=self._pass_op_options,
                                    sub_op_deps={"PythonBaseOp": [self._py_base_r.name]})
        return PyTorchInferOp(impl=py_base_op)
        # return py_base_op


class TorchModuleMixin:
    def __init__(self, *args, **kwargs) -> None:
        self._op = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._op is None:
            self._op = PytorchModule(
                self,
                trace_func=self._trace_func,
                *self._script_args,
                **self._script_kwargs)
        return self._op.__call__(*args, **kwargs)
