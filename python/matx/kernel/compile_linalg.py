#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from matx.kernel.kernel_parser import KernelParser
from matx.kernel.codegen.graph_ir_printer import GraphIRPrinter
import ctypes
from matx.kernel.typing import PYTYPE_TO_C_TYPE
import matx.kernel.typing.utils as typing_utils
from collections import OrderedDict
import subprocess
import os
import time
import numpy as np


def nd_to_c(nd, nd_t):
    allocated_ptr = nd.ctypes.data_as(ctypes.POINTER(PYTYPE_TO_C_TYPE[nd_t.dtype]))
    aligned_ptr = nd.ctypes.data_as(ctypes.POINTER(PYTYPE_TO_C_TYPE[nd_t.dtype]))
    offset = ctypes.c_int64(0)
    # shape = list(nd.ctypes.shape_as(c_int64))
    shape = [ctypes.c_int64(s) for s in nd.shape]
    strides = [ctypes.c_int64(s // nd.dtype.itemsize) for s in nd.strides]
    return [allocated_ptr, aligned_ptr, offset, *shape, *strides]


def scalar_to_c(v, v_t):
    v = PYTYPE_TO_C_TYPE[v_t.dtype](v)
    return v


def symbol_to_c(value):
    return ctypes.c_int64(value)


def bind_data_to_type(ins, types):
    args = []
    symbols = OrderedDict()
    for i, t in zip(ins, types):
        if not typing_utils.is_ndarray_type(t):
            raise NotImplementedError(f"{t} is not a legit type.")
        args.append((i, t))

        if typing_utils.is_scalar_type(t):
            continue
        for actual_s, annotated_s in zip(i.shape, t.shape):
            if not typing_utils.is_symbol(annotated_s):
                continue
            if annotated_s in symbols:
                assert symbols[annotated_s] == actual_s
                continue
            symbols[annotated_s] = actual_s
    return args, symbols


def binded_args_to_c(binded_args):
    args = []
    for value, t in binded_args:
        if typing_utils.is_scalar_type(t):
            args.append(scalar_to_c(value, t))
        elif typing_utils.is_ndarray_type(t):
            args += nd_to_c(value, t)
        elif typing_utils.is_symbol(t):
            args.append(symbol_to_c(value))
        else:
            raise NotImplementedError(f"{t} is not a legit type.")
    return args


def write_linalg(graph_ir, output_fname="tmp.mlir", debug=False, over_written_code=None):
    printer = GraphIRPrinter(graph_ir)
    code = printer.as_linalg_text()
    with open(output_fname, "w+") as f:
        if debug and over_written_code is not None:
            f.write(over_written_code)
        else:
            f.write(code)
    return output_fname


def lower_linalg_to_cpu(input_fname, output_fname="llvm_tmp.mlir"):
    env = os.environ.copy()
    lower = subprocess.Popen(['mlir-opt',
                              '--convert-linalg-to-loops',
                              '--lower-affine',
                              '--arith-expand',
                              '--arith-unsigned-when-equivalent',
                              '--convert-scf-to-cf',
                              '--convert-linalg-to-llvm',
                              '--convert-func-to-llvm',
                              '--convert-index-to-llvm',
                              '--convert-arith-to-llvm',
                              '--convert-memref-to-llvm',
                              '--convert-math-to-llvm',
                              '--convert-cf-to-llvm',
                              '--scf-for-loop-peeling',
                              '--scf-for-loop-specialization',
                              '--reconcile-unrealized-casts',
                              input_fname,
                              '-o',
                              output_fname],
                             env=env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    stdout, stderr = lower.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)
    return output_fname


def translate_to_llvm(input_fname, output_fname="llvm_tmp.ll"):
    env = os.environ.copy()
    to_llvm = subprocess.Popen(['mlir-translate',
                                '--mlir-to-llvmir',
                                input_fname,
                                '-o',
                                output_fname],
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = to_llvm.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)
    return output_fname


def llvm_compile(input_fname, output_fname="llvm_tmp.ll"):
    env = os.environ.copy()
    compile_llvm = subprocess.Popen(["llc",
                                     "-O3",
                                     "-filetype=obj",
                                     input_fname,
                                     "-o",
                                     input_fname + ".o"],
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    stdout, stderr = compile_llvm.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)

    compile_llvm = subprocess.Popen(["g++",
                                     "-shared",
                                     "-fPIC",
                                     "-o",
                                     output_fname,
                                     input_fname + ".o"],
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    stdout, stderr = compile_llvm.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)
    return output_fname


class LinalgFuncWrapper:

    def __init__(self, func, parser: KernelParser):
        self.func = func
        self.arg_types = parser.arg_types
        self.rt_types = parser.return_types
        if typing_utils.is_scalar_type(self.rt_types):
            self.func.restype = PYTYPE_TO_C_TYPE[self.rt_types.dtype]

    def __call__(self, *args, rt=None):
        if len(args) != len(self.arg_types):
            raise NotImplementedError(f"the size of the given input {len(args)}"
                                      f" is not the same as the annotation {len(self.arg_types)}")
        args, rt = self.to_c_args(*args, rt=rt)
        rc = self.raw_call(*args)
        return rt if rt is not None else rc

    def raw_call(self, *args):
        return self.func(*args)

    def to_c_args(self, *args, rt=None):
        binded_args, symbol_dict = bind_data_to_type(args, self.arg_types)
        if not typing_utils.is_scalar_type(self.rt_types):
            if rt is None:
                shape = [symbol_dict[s] if typing_utils.is_symbol(
                    s) else s for s in self.rt_types.shape]
                rt = np.zeros(shape=shape, dtype=self.rt_types.dtype)
            for actual_s, ann_s in zip(rt.shape, self.rt_types.shape):
                assert symbol_dict[ann_s] == actual_s
            binded_args.append((rt, self.rt_types))

        for t, value in symbol_dict.items():
            binded_args.append((value, t))
        return binded_args_to_c(binded_args), rt


def load_func(shared_lib, parser: KernelParser):
    linalg_func = ctypes.CDLL(os.path.join(os.getcwd(), shared_lib))
    func = getattr(linalg_func, parser.func_name)
    return LinalgFuncWrapper(func, parser)


def compile_linalg(
        parser: KernelParser,
        file_name=None,
        dir="__mlir_output__",
        debug=False,
        over_written_code=None):
    current_path = os.getcwd()
    try:
        if file_name is None:
            code_file_name = parser.file_name.split('/')[-1].split('.')[0]
            file_name = f"_{code_file_name}___{parser.func_name}_{int(time.time() * 100000)}"
        if debug:
            file_name = f"_mlir_debug"
        if not os.path.exists(dir):
            os.makedirs(dir)
        os.chdir(dir)
        mlir_f = write_linalg(parser.graph, file_name + ".mlir", debug, over_written_code)
        lowered_f = lower_linalg_to_cpu(mlir_f, "llvm_" + file_name + ".mlir")
        llvm_f = translate_to_llvm(lowered_f, "llvm_" + file_name + ".ll")
        shared_lib = llvm_compile(llvm_f, file_name + ".so")
        func = load_func(shared_lib, parser)
        return func
    except Exception as e:
        raise e
    finally:
        os.chdir(current_path)
