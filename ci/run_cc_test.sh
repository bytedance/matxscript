#!/usr/bin/env bash
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

THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

ROOT_PATH=${THIS_PATH}/../
BUILD_PATH=${ROOT_PATH}/lib
THIRD_PATH=${ROOT_PATH}/3rdparty

cd "${ROOT_PATH}" || exit 1

# mkdir lib
if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
else
    rm -rf "${BUILD_PATH:?}/*"
fi

# init submodule
cd "${ROOT_PATH}"
git submodule init
git submodule update --recursive

# build pcre
cd "${THIS_PATH}"
bash build_pcre.sh "${BUILD_PATH}/pcre"

# build matx
cd "${BUILD_PATH}"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_RUNTIME_STRING_INVARIANT_CHECK=ON ../
make -j8

cd "${BUILD_PATH}" || exit 1
./test/cc/cc_unit_test
