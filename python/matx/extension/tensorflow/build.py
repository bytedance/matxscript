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
import re
import copy
import subprocess
import tempfile
import tensorflow as tf

MATX_USER_DIR = os.environ.get('MATX_USER_DIR', os.path.expanduser('~/.matxscript/'))
try:
    os.makedirs(MATX_USER_DIR, exist_ok=True)
except:
    print('[WARNING] User directory created failed: ', MATX_USER_DIR, file=sys.stderr)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# for installation by pip
MATX_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
CMAKE_DIR = os.path.join(MATX_DIR, "extension/cpp/tensorflow")
if not os.path.exists(os.path.join(MATX_DIR, 'include')):
    # for development
    MATX_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../../"))
    CMAKE_DIR = os.path.join(MATX_DIR, "python/matx/extension/cpp/tensorflow")
if sys.platform.startswith('win32'):
    MATX_TF_LIB_NAME = 'libmatx_tensorflow+tensorflow{}.dll'.format(tf.__version__)
elif sys.platform.startswith('darwin'):
    MATX_TF_LIB_NAME = 'libmatx_tensorflow+tensorflow{}.dylib'.format(tf.__version__)
else:
    MATX_TF_LIB_NAME = 'libmatx_tensorflow+tensorflow{}.so'.format(tf.__version__)


def get_cmake_version():
    output = subprocess.check_output(['cmake', '--version']).decode('utf-8')
    line = output.splitlines()[0]
    version = line.split()[2]
    return version


def build_with_cmake():
    try:
        cmake_version = get_cmake_version()
        print("cmake_version: ", cmake_version)
    except:
        raise RuntimeError(
            "cmake not found by MATXScript TensorFlow extension, please install it!!!"
        ) from None
    tx_module = sys.modules['matx']
    tx_compile_flags = tx_module.cpp_extension.get_cflags()
    tx_lib_paths = tx_module.cpp_extension.library_paths()
    TX_CXX11_ABI_FLAG = tx_module.cpp_extension.USE_CXX11_ABI()
    TF_CXX11_ABI_FLAG = tf.sysconfig.CXX11_ABI_FLAG
    if TX_CXX11_ABI_FLAG != TF_CXX11_ABI_FLAG:
        raise RuntimeError(
            f"TensorFlow define _GLIBCXX_USE_CXX11_ABI={TF_CXX11_ABI_FLAG}, "
            f"but MATXScript define _GLIBCXX_USE_CXX11_ABI={TX_CXX11_ABI_FLAG}"
        ) from None
    tf_compile_flags = tf.sysconfig.get_compile_flags()
    all_compile_flags = copy.copy(tf_compile_flags)
    for flag in tx_compile_flags:
        flag = flag.strip()
        if flag not in all_compile_flags:
            all_compile_flags.append(flag)

    all_lib_names = ['matx', 'pcre']
    all_lib_paths = copy.copy(tx_lib_paths)
    tf_link_flags = tf.sysconfig.get_link_flags()
    for flag in tf_link_flags:
        flag = flag.strip()
        if flag.startswith('-L'):
            all_lib_paths.append(flag[2:])
        elif flag.startswith('-l'):
            all_lib_names.append(flag[2:])
        else:
            all_compile_flags.append(flag)
    print("all_compile_flags: ", all_compile_flags)
    print("all_lib_paths: ", all_lib_paths)
    print("all_lib_names: ", all_lib_names)
    all_compile_flags = ' '.join(all_compile_flags)
    all_lib_paths = ';'.join(all_lib_paths)
    all_lib_names = ';'.join(all_lib_names)
    curdir = os.getcwd()
    build_dir = tempfile.TemporaryDirectory(prefix="matxscript_tensorflow_build")
    print(f"[BUILD DIRECTORY]: {build_dir}")
    os.chdir(build_dir.name)
    cmake_cmd = f'''
    cmake \
    -DCMAKE_MATX_TF_COMPILE_FLAGS="{all_compile_flags}" \
    -DCMAKE_MATX_TF_LIB_PATHS="{all_lib_paths}" \
    -DCMAKE_MATX_TF_LIB_NAMES="{all_lib_names}" \
    -DCMAKE_TENSORFLOW_VERSION="{tf.__version__}" \
    {CMAKE_DIR}
    '''
    try:
        ret = os.system(cmake_cmd)
        assert ret == 0, "Failed to execute with cmake."
        ret = os.system('make VERBOSE=1 -j4')
        os.system(f'ls -l --color {build_dir.name}')
        errmsg = 'internal error: build libmatx_tensorflow failed!!!'
        assert ret == 0 and os.path.exists(MATX_TF_LIB_NAME), errmsg
        cp_cmd = 'cp {} {}'.format(MATX_TF_LIB_NAME, MATX_USER_DIR)
        ret = os.system(cp_cmd)
        assert ret == 0, 'failed to execute: {}'.format(cp_cmd)
    finally:
        os.chdir(curdir)
        build_dir.cleanup()


if __name__ == "__main__":
    build_with_cmake()
