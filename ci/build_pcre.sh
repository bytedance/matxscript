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

PCRE_INSTALL_PATH=$1

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/..
THIRD_PATH=${ROOT_PATH}/3rdparty
PCRE_SOURCE_PATH=${THIRD_PATH}/pcre-8.45
PCRE_BUILD_PATH=${THIRD_PATH}/pcre_build_845

mkdir -p "${PCRE_BUILD_PATH}"

# build pcre
cd "${PCRE_BUILD_PATH}"
echo > ChangeLog
"${PCRE_SOURCE_PATH}"/configure \
  --prefix="${PCRE_INSTALL_PATH}" \
  --enable-shared=yes \
  --enable-static=yes \
  --enable-utf \
  --enable-pcre16 \
  --enable-pcre32 \
  --enable-jit \
  --enable-unicode-properties \
  --enable-newline-is-lf \
  --with-pic=yes \
  --with-link-size=4

echo > ChangeLog
make -j 8
make install

# build pcre
#PCRE_CMAKE_FLAGS=" \
#-DCMAKE_INSTALL_PREFIX=${PCRE_INSTALL_PATH} \
#-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#-DBUILD_SHARED_LIBS=ON \
#-DBUILD_STATIC_LIBS=ON \
#-DPCRE_BUILD_PCRE8=ON \
#-DPCRE_SUPPORT_UTF=ON \
#-DPCRE_BUILD_PCRE16=ON \
#-DPCRE_BUILD_PCRE32=ON \
#-DPCRE_SUPPORT_JIT=ON \
#-DPCRE_SUPPORT_UNICODE_PROPERTIES=ON \
#-DPCRE_LINK_SIZE=4 \
#-DPCRE_NEWLINE=LF
#"
#mkdir -p "${PCRE_BUILD_PATH}"
#cd "${PCRE_BUILD_PATH}"
#cmake "${PCRE_CMAKE_FLAGS}" "${PCRE_SOURCE_PATH}"
#make -j$(nproc)
#make install
