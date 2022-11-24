# -*- coding: utf-8 -*-

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
import ctypes
import traceback
from ._libinfo import find_lib_path


def load_bundled_lib(libname):
    """Load bundled lib

    Parameters
    ----------
    libname : str
        lib file name, for example "libcut"

    Returns
    -------
    dll_handle : ctypes.CDLL

    """
    dll_path = find_lib_path(libname)
    lib_pp = os.path.abspath(os.path.dirname(dll_path[0]))
    cwd = os.getcwd()
    os.chdir(lib_pp)
    try:
        return ctypes.CDLL(dll_path[0], ctypes.RTLD_LOCAL)
    finally:
        os.chdir(cwd)
