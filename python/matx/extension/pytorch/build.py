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
import tempfile
import torch
import subprocess


MATX_USER_DIR = os.environ.get('MATX_USER_DIR', os.path.expanduser('~/.matxscript/'))
try:
    os.makedirs(MATX_USER_DIR, exist_ok=True)
except:
    print('[WARNING] User directory created failed: ', MATX_USER_DIR, file=sys.stderr)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# for installation by pip
MATX_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
CMAKE_DIR = os.path.join(MATX_DIR, "extension/cpp/pytorch")
if not os.path.exists(os.path.join(MATX_DIR, 'include')):
    # for development
    MATX_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../../"))
    CMAKE_DIR = os.path.join(MATX_DIR, "python/matx/extension/cpp/pytorch")
# please change name if we need update lib force
if sys.platform.startswith('win32'):
    MATX_PT_LIB_NAME = 'libmatx_pytorch+torch{}.dll'.format(torch.__version__)
elif sys.platform.startswith('darwin'):
    MATX_PT_LIB_NAME = 'libmatx_pytorch+torch{}.dylib'.format(torch.__version__)
else:
    MATX_PT_LIB_NAME = 'libmatx_pytorch+torch{}.so'.format(torch.__version__)


def get_cmake_version():
    output = subprocess.check_output(['cmake', '--version']).decode('utf-8')
    line = output.splitlines()[0]
    version = line.split()[2]
    return version


def build_with_cmake():
    try:
        cmake_version = get_cmake_version()
    except:
        raise RuntimeError(
            "cmake not found by matxscript PyTorch extension, please install it!!!") from None
    tx_module = sys.modules['matx']
    tx_compile_flags = ' '.join(tx_module.get_cflags())
    tx_link_flags = ' '.join(tx_module.get_link_flags())
    curdir = os.getcwd()
    build_dir = tempfile.TemporaryDirectory(prefix="matxscript_pytorch_build")
    print(f"[BUILD DIRECTORY]: {build_dir}")
    os.chdir(build_dir.name)
    from torch.utils import cpp_extension
    torch_incs = ""
    torch_lib_dir = ""
    ok_to_use_cuda = cpp_extension._find_cuda_home() is not None and torch.cuda.is_available()
    if ok_to_use_cuda:
        torch_incs = ';'.join(cpp_extension.include_paths(True))
        torch_lib_dir = ';'.join(cpp_extension.library_paths(True))
    else:
        torch_incs = ';'.join(cpp_extension.include_paths())
        torch_lib_dir = ';'.join(cpp_extension.library_paths())

    cmake_cmd = f'''
    cmake \
    -DCMAKE_TORCH_INCLUDES=\"{torch_incs}\" \
    -DCMAKE_TORCH_LIB=\"{torch_lib_dir}\" \
    -DCMAKE_MATX_COMPILE_FLAGS="{tx_compile_flags}" \
    -DCMAKE_MATX_LINK_FLAGS="{tx_link_flags}" \
    -DCMAKE_TORCH_VERSION="{torch.__version__}" \
    -DCMAKE_TORCH_CUDA_AVAILABLE={ok_to_use_cuda} \
    {CMAKE_DIR}
    '''

    try:
        ret = os.system(cmake_cmd)
        assert ret == 0, "Failed to execute with cmake."
        ret = os.system('make -j4')
        assert ret == 0 and os.path.exists(
            MATX_PT_LIB_NAME), 'internal error: build libmatx_pytorch failed!!!'
        cp_cmd = 'cp {} {}'.format(MATX_PT_LIB_NAME, MATX_USER_DIR)
        ret = os.system(cp_cmd)
        assert ret == 0, 'failed to execute: {}'.format(cp_cmd)
    finally:
        os.chdir(curdir)
        build_dir.cleanup()


if __name__ == "__main__":
    build_with_cmake()
