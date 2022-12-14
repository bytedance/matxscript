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

find_opencv=$(pkg-config --modversion opencv | grep -Eo "([0-9].[0-9].[0-9])")
echo $find_opencv
if [ "x${find_opencv}" != "x3.4.8" ]; then
    echo "no opencv found/ version is not 3.4.8"
    exit 0
fi
echo "found opencv 3.4.8. continue"

VISION_INSTALL_PATH=$1

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/..
VISION_CODE_PATH=${ROOT_PATH}/vision
BUILD_PATH=${VISION_CODE_PATH}/build
OUTPUT_PATH=${ROOT_PATH}/output/matx/lib

# mkdir lib
if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
else
  rm -rf "${BUILD_PATH:?}/*"
fi

pushd ${BUILD_PATH}

cmake ..

make -j8
make install

popd

# mkdir lib
mkdir -p "${VISION_INSTALL_PATH}"

cp ${VISION_CODE_PATH}/*so* ${VISION_INSTALL_PATH}
cp ${VISION_CODE_PATH}/*so* ${OUTPUT_PATH}