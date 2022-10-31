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
import time
import errno
from sysconfig import get_paths
from matx.contrib import cpp_extension
from matx.contrib import cc

_SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


def _get_dso_path():
    if (sys.platform == "darwin"
            or sys.platform.startswith("linux")
            or sys.platform.startswith("freebsd")):
        output_dso = _SCRIPT_PATH + os.sep + "matx_script_api.so"
    elif sys.platform == "win32":
        output_dso = _SCRIPT_PATH + os.sep + "matx_script_api.dll"
    else:
        raise ValueError("Unsupported platform")

    return output_dso


def _compile_fast_py_api(output_dso):
    py_include_path = get_paths()["include"]
    tx_cflags = cpp_extension.get_cflags()
    tx_cflags.append(f"-I{py_include_path}")
    options = ["-std=c++11"]
    if (sys.platform == "darwin"
            or sys.platform.startswith("linux")
            or sys.platform.startswith("freebsd")):
        options.append("-O2")
        options.append("-g")
    elif sys.platform == "win32":
        pass
    else:
        raise ValueError("Unsupported platform")

    options.extend(tx_cflags)

    objects = [_SCRIPT_PATH + os.sep + "fast_c_api.cc"]
    cc.create_shared(output_dso, objects, options)


def _load_matx_script_api_module(dso_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("matx_script_api", dso_path)
    _matx_script_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_matx_script_api)

    return _matx_script_api


def load_matx_script_api_module():
    RETRY_TIMES = 50
    WAIT = 0.2

    dso_path = _get_dso_path()
    lockfile_path = dso_path + '.lock'

    while RETRY_TIMES > 0:
        RETRY_TIMES -= 1

        try:
            _matx_script_api = _load_matx_script_api_module(dso_path)
            return _matx_script_api
        except ImportError as e:
            # No matx_script_api found, need compiling
            if not e.args or 'matx_script_api' not in e.args[0]:
                raise

            # Try to fetch a lock for compiling
            try:
                lockfile = os.open(lockfile_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

                time.sleep(WAIT)
                continue
            try:
                _compile_fast_py_api(dso_path)
            finally:
                os.close(lockfile)
                os.unlink(lockfile_path)

    raise ImportError


matx_script_api = load_matx_script_api_module()
