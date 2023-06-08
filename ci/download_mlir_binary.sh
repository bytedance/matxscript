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

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
MLIR_PATH=${ROOT_PATH}/mlir_build

if [ ! -d "$MLIR_PATH" ]; then
    mkdir "$MLIR_PATH"
fi

cd "$MLIR_PATH"


if [ ! -d "./pre_build" ]; then
    if [ ! -f "./clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz" ]; then
        wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    fi
    tar -xf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
    mv ./clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04 pre_build
fi


#export PATH=$PATH:/path/to/your/binarys/directory