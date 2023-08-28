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
from matx.kernel.codegen.graph_ir_printer import GraphIRPrinter
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.codegen.cpp_template.function_meta_data import get_codegen_data

interface_lib = set()

from matx._ffi.libinfo import find_lib_path

matx_lib_path = os.path.dirname(find_lib_path()[0])
os.environ['LD_LIBRARY_PATH'] = f'matx_lib_path:' + os.environ.get('LD_LIBRARY_PATH', '')


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
                              '--expand-strided-metadata',  # for lower strided memref
                              '--convert-linalg-to-loops',
                              '--lower-affine',
                              '--arith-expand',
                              '--memref-expand',
                              # '--normalize-memrefs',
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
                              # '--affine-expand-index-ops',
                              # '--affine-data-copy-generate',
                              # '--lower-affine',
                              # '--convert-arith-to-llvm',
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


def generate_matx_c_interface(parser, file_name, shard_lib_path):
    env = os.environ.copy()
    c_interface_code = get_codegen_data(
        parser, os.path.join(
            os.path.abspath(
                os.curdir), shard_lib_path))
    shard_lib_dir = os.path.abspath(os.path.dirname(shard_lib_path))
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
    output_so = os.path.join(shard_lib_dir, f"{file_name}_c_interface.so")
    compile_c_interface = subprocess.Popen(["g++",
                                            "-g",
                                            "-shared",
                                            "-o",
                                            f"{output_so}",
                                            output_file,
                                            f"-L{matx_lib_path}",
                                            "-lmatx"],
                                           env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    stdout, stderr = compile_c_interface.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n" + err)

    print(output_so)
    print(os.path.abspath('.'))
    print([f for f in os.listdir('.')])
    LIB = ctypes.CDLL(output_so, ctypes.RTLD_LOCAL)
    interface_lib.add(LIB)

    from .matx_compatible_interface import get_kernel_func
    func_name = "_" + str(c_interface_code.unique_id) + "_" + \
                c_interface_code.func_name + "__matx_c_api_"
    return get_kernel_func(func_name)


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
    except Exception as e:
        raise e
    finally:
        os.chdir(current_path)
