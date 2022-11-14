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
export USE_LIBBACKTRACE=OFF

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
BUILD_PATH=${ROOT_PATH}/android_build

OUTPUT_PATH=${ROOT_PATH}/android_output

# mkdir build path
if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
else
  rm -rf "${BUILD_PATH:?}/*"
fi

# mkdir output
if [ -d "${OUTPUT_PATH}" ]; then
    rm -rf "${OUTPUT_PATH}"
fi
mkdir -p "${OUTPUT_PATH}"

# init submodule
cd "${ROOT_PATH}"
git submodule init
git submodule update --recursive

# build matx
cd "${BUILD_PATH}"

cmake ../ \
-DBUILD_MICRO_RUNTIME=ON \
-DCMAKE_INSTALL_PREFIX="${OUTPUT_PATH}" \
-DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}"/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_shared \
-DANDROID_TOOLCHAIN=clang \
-DANDROID_NATIVE_API_LEVEL=android-21

make -j 8
make install
