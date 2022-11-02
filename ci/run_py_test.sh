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

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../

###############################################################################
# build all shared target
###############################################################################
cd "${ROOT_PATH}" || exit 1
BUILD_TESTING=OFF BUILD_BENCHMARK=OFF CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" bash ci/build_lib.sh

###############################################################################
# install requirements
###############################################################################
PYTHON_MODULE_PATH=${ROOT_PATH}/python
cd "${PYTHON_MODULE_PATH}"
pip3 install -r requirements.txt

###############################################################################
# find all test script
###############################################################################
PYTHONPATH=${PYTHONPATH:-}
TEST_SCRIPT_PATH=${ROOT_PATH}/test/script
cd "${TEST_SCRIPT_PATH}"
# shellcheck disable=SC2045
for script_file in $(ls test_*.py); do
  echo "test script: ${script_file}"
  PYTHONPATH="${ROOT_PATH}/python:${PYTHONPATH}" python3 "${script_file}"
done
