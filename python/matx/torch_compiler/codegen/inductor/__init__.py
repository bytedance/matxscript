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

from typing import List, Tuple

import torch
import torch._inductor.compile_fx as compile_fx
from torch import fx
from torch._inductor.debug import DebugContext
from torch._inductor.virtualized import V

"""
Use a global variable to hack the compile_fx_inner and record the compiled code.
This works in single process problem, but requires careful review in multi-processing
"""


class FakeCallableWithCode():
    code = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def set_code(self, code):
        self.code = code


fake_callable = FakeCallableWithCode()


@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner_cpu(
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        cudagraphs=None,
        num_fixed=0,
        is_backward=False,
        graph_id=None,
):
    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    compile_fx.sys.setrecursionlimit(max(compile_fx.sys.getrecursionlimit(), 2000))

    V.debug.fx_graph(gm, example_inputs)

    shape_env = compile_fx._shape_env_from_inputs(example_inputs)
    fake_mode = compile_fx.fake_mode_from_tensors(
        example_inputs
    ) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

    with V.set_fake_mode(fake_mode):
        compile_fx.pattern_matcher.fx_passes(gm)
        V.debug.fx_graph_transformed(gm, example_inputs)

        graph = compile_fx.GraphLowering(
            gm,
            shape_env=shape_env,
            num_static_inputs=num_fixed,
            graph_id=graph_id,
        )
        with V.set_graph_handler(graph):
            graph.run(*example_inputs)
            code = graph.codegen()
            fake_callable.set_code(code)

    return fake_callable


def assert_tuple_of_tensors(tensors):
    assert isinstance(tensors, Tuple)
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor), 'Each element in tensors must be a torch.Tensor'


from torch._subclasses import FakeTensor, FakeTensorMode


def extract_inductor_code(kernel, example_inputs):
    # check kernel input and output. All the input must be a Tensor. The output must be a tuple of Tensor
    # TODO: remove this constraints (long term)
    assert isinstance(example_inputs, (List, Tuple))
    example_inputs = tuple(example_inputs)
    assert_tuple_of_tensors(example_inputs)
    fake_mode = FakeTensorMode()
    fake_example_inputs = [FakeTensor.from_tensor(t, fake_mode=fake_mode) for t in example_inputs]
    fake_output = kernel(*fake_example_inputs)
    assert_tuple_of_tensors(fake_output)

    model = fx.symbolic_trace(kernel)
    compile_fx.compile_fx(
        model,
        example_inputs_=fake_example_inputs,
        inner_compile=compile_fx_inner_cpu)

    code = fake_callable.code

    # By default, Pytorch compiles a Python module with all the C++ kernel with unified name kernel.
    # The actual kernel name should be kernel.__name__.
    # TODO: fix this after rewriting inductor codegen to all C++ instead of a Python module
    kernel_name = kernel.__name__

    # fake_output is used
    return code, kernel_name, fake_output
