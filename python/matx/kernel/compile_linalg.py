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

import ctypes
import os
import pathlib
import subprocess
import time
from collections import OrderedDict

import numpy as np

import matx
import matx.kernel.typing.utils as typing_utils
from matx.kernel.codegen.graph_ir_printer import GraphIRPrinter
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.typing import PYTYPE_TO_C_TYPE, STR_TO_PYTYPE
from matx.kernel.codegen.cpp_template.function_meta_data import from_kernel_parser


def make_memref_descriptor(
        dim,
        allocated_ptr=None,
        aligned_ptr=None,
        offset=0,
        shape=None,
        strides=None):
    if shape is None:
        shape = [0] * dim

    if strides is None:
        strides = [0] * dim

    shape_array_t = ctypes.c_int64 * dim
    stride_array_t = ctypes.c_int64 * dim

    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated_ptr", ctypes.c_void_p),
            ("aligned_ptr", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("shape", shape_array_t),
            ("strides", stride_array_t)
        ]

    memref_struct = MemRefDescriptor(allocated_ptr, aligned_ptr, offset,
                                     shape_array_t(*shape), stride_array_t(*strides))
    return memref_struct


def nd_to_c(nd):
    allocated_ptr = nd.ctypes.data_as(ctypes.c_void_p)
    aligned_ptr = nd.ctypes.data_as(ctypes.c_void_p)
    offset = 0
    # shape = list(nd.ctypes.shape_as(c_int64))]
    shape = [ctypes.c_int64(s) for s in nd.shape]
    strides = [ctypes.c_int64(s // nd.dtype.itemsize) for s in nd.strides]
    dim = len(shape)
    # return [allocated_ptr, aligned_ptr, offset, *shape, *strides]
    return make_memref_descriptor(dim, allocated_ptr, aligned_ptr, offset, shape, strides)


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
            args.append(ctypes.byref(nd_to_c(value)))
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
                              '--memref-expand',
                              '--normalize-memrefs',
                              '--fold-memref-alias-ops',
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
                              '--affine-expand-index-ops',
                              '--affine-data-copy-generate',
                              '--lower-affine',
                              '--convert-arith-to-llvm',
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


def _is_inputs(in_args, rt):
    if not hasattr(rt, "allocated_ptr"):
        return
    rt_allocated_ptr = getattr(rt, "allocated_ptr")
    for a in in_args:
        if not hasattr(a, "_obj"):
            continue
        a = a._obj
        if not hasattr(a, "allocated_ptr"):
            continue
        a_allocated_ptr = getattr(a, "allocated_ptr")
        if a_allocated_ptr == rt_allocated_ptr:
            return True
    return False


class LinalgFuncWrapper:

    def __init__(self, func, parser: KernelParser, deallocator):
        self.func = func
        self.deallocator = deallocator
        self.arg_types = parser.arg_types
        self.rt_types = parser.return_types
        self.func_return_kind = parser.graph.func_return_kind
        if self.func_return_kind.is_scalar():
            self.func.restype = PYTYPE_TO_C_TYPE[self.rt_types.dtype]
        self.return_dim = len(parser.graph.return_shape)
        self.return_dtype_str = parser.graph.return_dtype_str

    def __call__(self, *args, rt=None):
        if len(args) != len(self.arg_types):
            raise NotImplementedError(f"the size of the given input {len(args)}"
                                      f" is not the same as the annotation {len(self.arg_types)}")
        c_args, rt_np_nd = self.to_c_args(*args, rt_np_nd=rt)
        return self.call_c_arg(*c_args, rt_np_nd=rt_np_nd)

    def call_c_arg(self, *c_args, rt_np_nd=None):
        # if this is a void function do nothing
        if self.func_return_kind.is_void():
            self.raw_call(*c_args)
            return
        # if the return is a scalar, then just return the scalar
        elif self.func_return_kind.is_scalar():
            return self.raw_call(*c_args)
        # if the return is a static tensor, then return the pre allocated a np nd
        elif self.func_return_kind.is_static_tensor():
            self.raw_call(*c_args)
            return rt_np_nd
        # if the return is a dynamic tensor,
        elif self.func_return_kind.is_dynamic_tensor():
            # the memory of this memref is to be allocated by MLIR
            return_nd = make_memref_descriptor(self.return_dim)
            return_nd_ptr = ctypes.byref(return_nd)
            self.raw_call(*(return_nd_ptr, *c_args))
            # based on the return, construct a new array,
            # so the memory space is managed by python
            shape = return_nd.shape
            rt_dtype = PYTYPE_TO_C_TYPE[STR_TO_PYTYPE[self.return_dtype_str]]
            rt_ptr_type = ctypes.POINTER(rt_dtype)
            casted_return_ptr = ctypes.cast(
                return_nd.aligned_ptr + return_nd.offset, rt_ptr_type)
            rt = np.ctypeslib.as_array(casted_return_ptr, shape)
            rt = np.array(rt, copy=True)
            # if the memref ends up pointing to memory space allocated by MLIR
            # deallocate the space
            if not _is_inputs(c_args, return_nd):
                self.deallocator(return_nd.allocated_ptr)
            return rt
        else:
            raise SyntaxError(f"unknown function return kind {self.func_return_kind}")

    def raw_call(self, *args):
        return self.func(*args)

    def to_c_args(self, *args, rt_np_nd=None):
        binded_args, symbol_dict = bind_data_to_type(args, self.arg_types)
        if self.func_return_kind.is_static_tensor():
            if rt_np_nd is None:
                shape = [symbol_dict[s] if typing_utils.is_symbol(
                    s) else s for s in self.rt_types.shape]
                rt_np_nd = np.zeros(shape=shape, dtype=self.rt_types.dtype)
            binded_args.insert(0, (rt_np_nd, self.rt_types))
        for t, value in symbol_dict.items():
            binded_args.append((value, t))
        c_args = binded_args_to_c(binded_args)
        return c_args, rt_np_nd

    def _to_py_return(self, memref_descriptor):
        ...


def load_func(shared_lib, parser: KernelParser):
    linalg_func = ctypes.CDLL(os.path.join(os.getcwd(), shared_lib))
    func = getattr(linalg_func, f"_mlir_ciface_{parser.func_name}")
    deallocator = getattr(linalg_func, f"free")
    return LinalgFuncWrapper(func, parser, deallocator)


interface_lib = set()


def generate_matx_c_interface(parser, file_name, shard_lib_path):
    env = os.environ.copy()
    c_interface_code = from_kernel_parser(
        parser, os.path.join(
            os.path.abspath(
                os.curdir), shard_lib_path))
    shard_lib_dir = os.path.dirname(shard_lib_path)
    file_path = os.path.join(shard_lib_dir, f"{file_name}_c_interface.cpp")
    with open(file_path, "w+") as f:
        f.write(c_interface_code.code())

    include_dir = pathlib.Path(os.path.abspath(
        __file__)).parent.parent.parent.parent.joinpath("include")
    output_file = os.path.join(shard_lib_dir, f"{file_name}_c_interface.o")

    # gcc -fPIC -I/matxscript/include -std=c++14 file1.c
    compile_c_interface = subprocess.Popen(["g++",
                                            "-fPIC",
                                            "-g",
                                            f"-I{include_dir}",
                                            "-std=c++14",
                                            "-c",
                                            file_path],
                                           env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    stdout, stderr = compile_c_interface.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)

    # gcc -shared -o libexample.so file1.o file2.o
    compile_c_interface = subprocess.Popen(["g++",
                                            "-g",
                                            "-shared",
                                            "-o",
                                            f"lib{file_name}_c_interface.so",
                                            output_file,
                                            "-L/Users/bytedance/Desktop/workspace/matxscript/lib/",
                                            "-lmatx"],
                                           env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    stdout, stderr = compile_c_interface.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)

    LIB = ctypes.CDLL(
        os.path.join(
            shard_lib_dir,
            f"lib{file_name}_c_interface.so"),
        ctypes.RTLD_LOCAL)
    interface_lib.add(LIB)

    from .matx_compatible_interface import KernelFunction, get_kernel_func
    func_name = "_" + str(c_interface_code.unique_id) + "_" + \
        c_interface_code.func_name + "__matx_c_api_"
    return get_kernel_func(func_name)
    py_wrapper = KernelFunction("_" + str(c_interface_code.unique_id) +
                                "_" + c_interface_code.func_name + "__matx_c_api_")

    return py_wrapper


def compile_linalg(
        parser: KernelParser,
        file_name=None,
        out_dir="__mlir_output__",
        debug=False,
        over_written_code=None):
    current_path = os.getcwd()
    try:
        if file_name is None:
            code_file_name = parser.file_name.split('/')[-1].split('.')[0]
            file_name = f"_{code_file_name}___{parser.func_name}_{int(time.time() * 100000)}"
        if debug:
            file_name = f"_mlir_debug"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        os.chdir(out_dir)
        # codegen linalg code to local file
        mlir_f = write_linalg(parser.graph, file_name + ".mlir", debug, over_written_code)
        # apply mlir passes
        lowered_f = lower_linalg_to_cpu(mlir_f, "llvm_" + file_name + ".mlir")
        # lower mlir to llvm
        llvm_f = translate_to_llvm(lowered_f, "llvm_" + file_name + ".ll")
        # compile llvm code to shared library
        shared_lib = llvm_compile(llvm_f, file_name + ".so")
        # codegen the c inter face that is compatible with matx
        matx_c_interface = generate_matx_c_interface(parser, file_name, shared_lib)
        return matx_c_interface

        # todo remove this part
        func = load_func(shared_lib, parser)
        return func
    except Exception as e:
        raise e
    finally:
        os.chdir(current_path)
