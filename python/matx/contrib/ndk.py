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
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Tuple


# this function is needed for python3
# to convert ctypes.char_p .value back to python str
if sys.platform == "win32":
    def _py_str(x):
        try:
            return x.decode('utf-8')
        except UnicodeDecodeError:
            encoding = 'cp' + str(ctypes.cdll.kernel32.GetACP())
        return x.decode(encoding)

    py_str = _py_str
else:
    py_str = lambda x: x.decode('utf-8')


CMAKELISTS_TEMPLATE = """cmake_minimum_required(VERSION 3.10)
project(nkd_build)
message(STATUS ${{CMAKE_C_COMPILER}})
message(STATUS ${{CMAKE_CXX_COMPILER}})
set(CMAKE_C_LINK_FLAGS)
set(CMAKE_CXX_LINK_FLAGS)
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -g -O3 -std=c++14 {cxx_flags}")
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY {output_directory})
include_directories({include_directories})
link_directories({link_directories})
add_library({target} SHARED {source})
SET_TARGET_PROPERTIES({target} PROPERTIES LINK_FLAGS_RELEASE -s)
target_link_libraries({target} matx {link_libraries})"""


def format_argument(argument):
    if argument is None:
        return ''
    elif isinstance(argument, str):
        return argument
    elif isinstance(argument, (List, Tuple)):
        return ' '.join(argument)


def create_shared(
        file_path,
        include_paths=None,
        cmake_arguments=None,
        cxx_flags=None,
        link_directories=None,
        link_libraries=None):

    output_directory = os.path.dirname(file_path)
    target = os.path.basename(file_path)[3:-3]
    cmakelists_context = CMAKELISTS_TEMPLATE.format(
        target=target,
        output_directory=output_directory,
        include_directories=format_argument(include_paths),
        link_directories=format_argument(link_directories),
        source=file_path,
        link_libraries=format_argument(link_libraries),
        cxx_flags=format_argument(cxx_flags)
    )
    with TemporaryDirectory() as tmpdir:
        cmakelists_path = tmpdir + os.sep + "CMakeLists.txt"
        with open(cmakelists_path, 'w') as cmakelists_file:
            cmakelists_file.write(cmakelists_context)
        cmd = ["cmake ."]
        cmd += cmake_arguments
        cmd += ["&&"]
        cmd += ["make"]
        cmd = ' '.join(cmd)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=tmpdir,
            shell=True)
        if proc.returncode != 0:
            msg = "CMakeLists.txt: \n" + cmakelists_context
            msg += "Compilation error:\n"
            if proc.stdout:
                msg += py_str(proc.stdout)
            if proc.stderr:
                msg += py_str(proc.stderr)

            raise RuntimeError(msg)
