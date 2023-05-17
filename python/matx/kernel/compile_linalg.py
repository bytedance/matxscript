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

from matx.ir import _ffi_node_api
from .kernel_parser import KernelParser
from ctypes import *
from .typing import *
from collections import OrderedDict
from itertools import chain
import subprocess
import os
import time


def compile_linalg(matx_ir, file_name="tmp"):
    code = _ffi_node_api.as_linalg_text(matx_ir).decode()
    with open(file_name + ".mlir", "w+") as f:
        f.write(code)
    env = os.environ.copy()
    lower = subprocess.Popen(['mlir-opt',
                              '--convert-linalg-to-loops',
                              '--lower-affine',
                              '--convert-scf-to-cf',
                              '--convert-linalg-to-llvm',
                              '--convert-func-to-llvm',
                              '--convert-index-to-llvm',
                              '--convert-arith-to-llvm',
                              '--convert-memref-to-llvm',
                              '--convert-cf-to-llvm',
                              '--reconcile-unrealized-casts',
                              file_name + ".mlir",
                              '-o',
                              'llvm_' + file_name + ".mlir"],
                             env=env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    stdout, stderr = lower.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)
    to_llvm = subprocess.Popen(['mlir-translate',
                                '--mlir-to-llvmir',
                                'llvm_' + file_name + ".mlir",
                                '-o',
                                'llvm_' + file_name + ".ll"],
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = to_llvm.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)
    compile_llvm = subprocess.Popen(["llc",
                                     "-filetype=obj",
                                     'llvm_' + file_name + ".ll",
                                     "-o",
                                     file_name + ".o"],
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
                                     file_name + ".so",
                                     file_name + ".o"],
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    stdout, stderr = compile_llvm.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)


def nd_to_c(nd, nd_t):
    allocated_ptr = nd.ctypes.data_as(POINTER(PYTYPE_TO_C_TYPE[nd_t.dtype]))
    aligned_ptr = nd.ctypes.data_as(POINTER(PYTYPE_TO_C_TYPE[nd_t.dtype]))
    offset = c_int64(0)
    shape = list(nd.ctypes.shape_as(c_int64))
    print(", ".join([f"shape{i} = {shape[i]}" for i in range(len(shape))]))
    print(nd.strides)
    strides = [c_int64(s // nd.dtype.itemsize) for s in nd.strides]
    print(", ".join([f"strides{i} = {strides[i]}" for i in range(len(strides))]))
    return [allocated_ptr, aligned_ptr, offset, *shape, *strides]


def scalar_to_c(v, v_t):
    v = PYTYPE_TO_C_TYPE[v_t.dtype](v)
    return v

def symbol_to_c(value):
    print(f"symbol = {value}")
    return c_int64(value)


def bind_data_to_type(ins, types):
    args = []
    symbols = OrderedDict()
    for i, t in zip(ins, types):
        if not is_ndarray_type(t):
            raise NotImplementedError(f"{t} is not a legit type.")
        args.append((i, t))

        for actual_s, annotated_s in zip(i.shape, t.shape):
            if not is_symbol(annotated_s):
                continue
            if annotated_s in symbols:
                assert symbols[annotated_s] == actual_s
                continue
            symbols[annotated_s] = actual_s
    return args, symbols


def binded_args_to_c(binded_args):
    args = []
    for value, t in binded_args:
        if is_scalar_type(t):
            args.append(scalar_to_c(value, t))
        elif is_ndarray_type(t):
            args += nd_to_c(value, t)
        elif is_symbol(t):
            args.append(symbol_to_c(value))
        else:
            raise NotImplementedError(f"{t} is not a legit type.")
    return args


def to_c_args(parser: KernelParser, ins, rt=None):
    args_types = parser.arg_types
    rt_types = parser.return_types
    binded_args, symbol_dict = bind_data_to_type(ins, args_types)
    if rt is None:  # todo shape may be symbol
        shape = [symbol_dict[s] if is_symbol(s) else s for s in rt_types.shape]
        rt = np.zeros(shape=shape, dtype=rt_types.dtype)
    for actual_s, ann_s in zip(rt.shape, rt_types.shape):
        assert symbol_dict[ann_s] == actual_s
    binded_args.append((rt, rt_types))
    for t, value in symbol_dict.items():
        binded_args.append((value, t))
    return binded_args_to_c(binded_args), rt


def run(parser: KernelParser, *args, rt=None):
    if len(args) != len(parser.arg_types):
        raise NotImplementedError(f"the size of the given input {len(args)}"
                                  f" is not the same as the annotation {len(parser.arg_types)}")
    code_file_name = parser.file_name.split('/')[-1].split('.')[0]
    file_name = f"_{code_file_name}___{parser.func_name}_{int(time.time() * 100000)}"
    print(file_name)
    compile_linalg(parser.main_node_ir, file_name)
    linalg_func = CDLL(file_name + ".so")
    func = getattr(linalg_func, parser.func_name)
    args, rt = to_c_args(parser, args, rt=rt)
    print(args)
    func(*args)
    print("pass")
    print(rt)
    return rt
