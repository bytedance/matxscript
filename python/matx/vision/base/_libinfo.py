# -*- coding:utf-8 -*-

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

from __future__ import absolute_import
import os
import sys


def find_lib_path(lib_name):
    base_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    base_path = os.path.join(base_path, '../')
    api_path = os.path.join(base_path, 'lib/')
    matx_build_path = os.path.join(base_path, '../../../vision')
    python_lib_path = os.path.join(base_path, '../lib')
    dll_path = [base_path, api_path, matx_build_path, python_lib_path]
    if sys.platform.startswith('win32'):
        dll_path = [os.path.join(p, '%s.dll' % lib_name) for p in dll_path]
    elif sys.platform.startswith('darwin'):
        dll_path = [os.path.join(p, '%s.dylib' % lib_name) for p in dll_path]
    else:
        dll_path = [os.path.join(p, '%s.so' % lib_name) for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise FileNotFoundError('Cannot find the lib: %s.\n' % lib_name +
                                'List of search path candidates:\n' + str('\n'.join(dll_path)))
    return lib_path
