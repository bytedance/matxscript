# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Module is inspired by incubator-tvm.
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


from . import _ffi_api

FATAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DEBUG = 10


def set_cpp_logging_level(logging_level):
    """

    Parameters
    ----------
    logging_level : int
        Set the logging level of CPP

    Returns
    -------

    """
    _ffi_api.SetLoggingLevel(logging_level)


def get_cpp_logging_level():
    return _ffi_api.GetLoggingLevel()
