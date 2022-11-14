#! /usr/bin/env bash
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

set -xue
set -o pipefail

export BUILD_TESTING=OFF
export BUILD_BENCHMARK=OFF
export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

ROOT_PATH=${THIS_PATH}/..
BUILD_PATH=${ROOT_PATH}/lib
THIRD_PATH=${ROOT_PATH}/3rdparty
OUTPUT_PATH=${ROOT_PATH}/output
OUTPUT_LIB_PATH=${OUTPUT_PATH}/matx/lib

function set_build_info() {
  local branch
  local short_commit_id
  local init_py_file=$1
  branch=$(git rev-parse --abbrev-ref HEAD)
  short_commit_id=$(git rev-parse --short HEAD)

  if [ "$(uname)" == "Darwin" ]; then
    sed -i -e "s:__branch__ = None:__branch__ = '${branch}':g" "${init_py_file}"
    sed -i -e "s:__commit_id__ = None:__commit_id__ = '${short_commit_id}':g" "${init_py_file}"
  else
    sed -i "s:__branch__ = None:__branch__ = '${branch}':g" "${init_py_file}"
    sed -i "s:__commit_id__ = None:__commit_id__ = '${short_commit_id}':g" "${init_py_file}"
  fi
}

function set_cuda_home() {
  DEFAULT_CUDA_HOME=/usr/local/cuda/
  if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=${CUDA_HOME}
  else
    export CUDA_HOME=${DEFAULT_CUDA_HOME}
  fi
  # check cuda valid or not
  "${CUDA_HOME}"/bin/nvcc --version || unset CUDA_HOME

  if [ -z "${CUDA_HOME:-}" ]; then
    return 0
  fi

  TOOLKIT_VERSION=$(${CUDA_HOME}/bin/nvcc --version | grep -Po "release \K([0-9]{1,}\.)+[0-9]{1,}" )
  echo ${TOOLKIT_VERSION}

  if [[ "x${TOOLKIT_VERSION}" == "x11.1" ]]; then
    echo "TOOLKIT_VERSION: ${TOOLKIT_VERSION}"
    export MLSYS_COMPILE_CUDA_VERSION=11.1
    export MLSYS_COMPILE_CUDNN_VERSION=8.2.0
  elif [[ "x${TOOLKIT_VERSION}" == "x11.3" ]]; then
    export MLSYS_COMPILE_CUDA_VERSION=11.3
    export MLSYS_COMPILE_CUDNN_VERSION=8.0.4
  fi
}


# cuda
set_cuda_home

# mkdir lib
if [ -d "${BUILD_PATH}" ]; then
    rm -rf "${BUILD_PATH}"
fi
mkdir -p "${BUILD_PATH}"

# mkdir output
if [ -d "${OUTPUT_PATH}" ]; then
    rm -rf "${OUTPUT_PATH}"
fi
mkdir -p "${OUTPUT_PATH}"

# init submodule
cd "${ROOT_PATH}"
git submodule init
git submodule update --recursive

# build pcre
cd "${THIS_PATH}"
bash build_pcre.sh "${BUILD_PATH}/pcre"

# build matx
cd "${BUILD_PATH}"

USE_LIBBACKTRACE=${USE_LIBBACKTRACE:-ON}
export USE_LIBBACKTRACE=${USE_LIBBACKTRACE}

cmake ../ -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}
make -j 8
make install

# prebuild matx_script_api.so
if [ "$(uname)" == "Darwin" ]; then
    cd ${OUTPUT_PATH} && python3 -c 'import matx'
else
    PYTHON_INC_PATH=$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"]);')
    g++ -std=c++11 -fPIC -shared -O2 -I"${PYTHON_INC_PATH}" -I"${ROOT_PATH}/include" -o "${OUTPUT_PATH}/matx/_ffi/_c_ext/matx_script_api.so" "${ROOT_PATH}/python/matx/_ffi/_c_ext/fast_c_api.cc"
fi

# set build info
set_build_info "${OUTPUT_PATH}/matx/__init__.py"

# clean __pycache__
find ${OUTPUT_PATH} -name "__pycache__" -type d -print
find ${OUTPUT_PATH} -name "__pycache__" -type d -print | xargs rm -rf

# install wheel
pip3 install wheel

# build whl
cd ${OUTPUT_PATH} && python3 setup.py clean
cd ${OUTPUT_PATH} && python3 setup.py bdist_wheel $@
