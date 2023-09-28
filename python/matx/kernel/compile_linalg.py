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
from matx.kernel.codegen.cpp_template import render_matx_api_code

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
    print(stderr.decode())
    if lower.returncode != 0:
        raise RuntimeError("\n" + f"Command failed with exit code {lower.returncode}")
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
    print(stderr.decode())
    if to_llvm.returncode != 0:
        raise RuntimeError("\n" + f"Command failed with exit code {to_llvm.returncode}")
    return output_fname


def llvm_compile(input_fname, output_fname="llvm_tmp.ll.o"):
    env = os.environ.copy()
    compile_llvm = subprocess.Popen(["llc",
                                     "-O3",
                                     "-filetype=obj",
                                     input_fname,
                                     "-o",
                                     output_fname],
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    stdout, stderr = compile_llvm.communicate()
    print(stdout.decode())
    print(stderr.decode())
    if compile_llvm.returncode != 0:
        raise RuntimeError("\n" + f"Command failed with exit code {compile_llvm.returncode}")
    return output_fname


def generate_matx_c_interface(parser, file_name, mlir_object_file):
    env = os.environ.copy()
    c_interface_code, meta_data = render_matx_api_code(parser)
    shard_lib_dir = os.path.abspath(os.path.dirname(mlir_object_file))
    file_path = os.path.join(shard_lib_dir, f"{file_name}_c_interface.cpp")
    with open(file_path, "w+") as f:
        f.write(c_interface_code)

    include_dir = pathlib.Path(os.path.abspath(
        __file__)).parent.parent.parent.parent.joinpath("include")
    output_file = os.path.join(shard_lib_dir, f"{file_name}_c_interface.o")

    # gcc -fPIC -I/matxscript/include -std=c++14 file1.c
    compile_c_interface = subprocess.Popen(["g++",
                                            "-fPIC",
                                            f"-I{include_dir}",
                                            "-std=c++14",
                                            "-c",
                                            file_path],
                                           env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    stdout, stderr = compile_c_interface.communicate()
    print(stdout.decode())
    print(stderr.decode())
    if compile_c_interface.returncode != 0:
        raise RuntimeError("\n" + f"Command failed with exit code {compile_c_interface.returncode}")

    # gcc -shared -o libexample.so file1.o file2.o
    output_so = os.path.join(shard_lib_dir, f"{file_name}_c_interface.so")
    compile_c_interface = subprocess.Popen(["g++",
                                            "-shared",
                                            "-o",
                                            output_so,
                                            output_file,
                                            mlir_object_file,
                                            f"-L{matx_lib_path}",
                                            "-lmatx"],
                                           env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    stdout, stderr = compile_c_interface.communicate()
    print(stdout.decode())
    print(stderr.decode())
    if compile_c_interface.returncode != 0:
        raise RuntimeError("\n" + f"Command failed with exit code {compile_c_interface.returncode}")

    print(output_so)
    print(os.path.abspath('.'))
    print([f for f in os.listdir('.')])

    from .matx_compatible_interface import get_kernel_func
    return get_kernel_func(
        output_so,
        meta_data.python_func_name,
        meta_data.file_name,
        meta_data.line_no)


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
        mlir_object_file = llvm_compile(llvm_f, file_name + ".o")
        # codegen the c inter face that is compatible with matx
        matx_c_interface = generate_matx_c_interface(parser, file_name, mlir_object_file)
        return matx_c_interface
    except Exception as e:
        raise e
    finally:
        os.chdir(current_path)
