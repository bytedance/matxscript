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
from .._ffi.libinfo import find_include_path, find_lib_path
from .._ffi.base import USE_CXX11_ABI


def get_json_inc_path():
    try:
        rapidjson_path = find_include_path("rapidjson")
    except:
        return None
    if not rapidjson_path or len(rapidjson_path) == 0:
        return None
    real_path = os.path.join(rapidjson_path[0], "include")
    if os.path.exists(real_path) and os.path.isdir(real_path):
        return real_path
    return None


def get_cflags():
    inc_paths = find_include_path()
    cflags = []
    assert inc_paths is not None and len(inc_paths) > 0
    cflags.append("-I" + inc_paths[0].strip())

    thirdparty_json_path = get_json_inc_path()
    if thirdparty_json_path:
        cflags.append("-I" + thirdparty_json_path)

    if USE_CXX11_ABI():
        cflags.append("-D_GLIBCXX_USE_CXX11_ABI=1")
    else:
        cflags.append("-D_GLIBCXX_USE_CXX11_ABI=0")

    return cflags


def get_link_flags():
    """ lib layout
    - lib/
      - libmatx.so
      - pcre/
        libpcre.so
    """
    ldflags = []
    ldflags.append("-ldl")
    ldflags.append("-lpthread")

    # ---------libmatx.so-------------------#
    lib_path = find_lib_path()
    libdir = os.path.dirname(lib_path[0])
    sofile = os.path.basename(lib_path[0])

    assert sofile.startswith("lib")
    soname = sofile.split('.')[0][3:]

    ldflags.append("-L" + libdir)
    ldflags.append("-l" + soname)

    # -------libpcre.so-------------------#
    if sys.platform.startswith('win32'):
        pcre_soname = "libpcre.dll"
    elif sys.platform.startswith('darwin'):
        pcre_soname = "libpcre.dylib"
    else:
        pcre_soname = "libpcre.so"
    pcrepath = find_lib_path(pcre_soname, [libdir, os.path.join(libdir, "pcre/lib")])
    pcre_dir = os.path.dirname(pcrepath[0])

    ldflags.append("-L" + pcre_dir)
    ldflags.append("-lpcre")

    return ldflags


def include_paths():
    incs = find_include_path()
    incs.append(get_json_inc_path())
    return incs


def library_paths():
    lib_paths = []
    # ---------libmatx.so-------------------#
    lib_location = find_lib_path()
    lib_dir = os.path.dirname(lib_location[0])
    lib_paths.append(lib_dir)

    # -------libpcre.so-------------------#
    if sys.platform.startswith('win32'):
        pcre_soname = "libpcre.dll"
    elif sys.platform.startswith('darwin'):
        pcre_soname = "libpcre.dylib"
    else:
        pcre_soname = "libpcre.so"
    pcre_lib_location = find_lib_path(pcre_soname, [lib_dir, os.path.join(lib_dir, "pcre/lib")])
    pcre_dir = os.path.dirname(pcre_lib_location[0])

    if pcre_dir != lib_paths:
        lib_paths.append(pcre_dir)

    return lib_paths
