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

THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

ROOT_PATH=${THIS_PATH}/..
OUTPUT_PATH=${ROOT_PATH}/output

bash ${THIS_PATH}/build_pip_whl.sh --python-tag py3 --plat-name macosx_10_9_x86_64

# cp dist
cp -r ${OUTPUT_PATH}/dist ./
