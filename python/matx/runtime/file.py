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

from io import UnsupportedOperation
from .. import _ffi
from .object import Object
from . import _ffi_api


@_ffi.register_object("runtime.File")
class File(Object):
    """A simple file class, which only supports reading by lines now.

    File(path, mode, encoding) -> similar to builtins.open
    """

    def __init__(self, path, mode='r', encoding='utf-8') -> None:
        self.__init_handle_by_constructor__(_ffi_api.File, path, mode, encoding)
        self.path = path
        self.mode = mode
        self.encoding = encoding

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def readline(self):
        if self.mode == 'r':
            return _ffi_api.FileReadLineUnicode(self)
        elif self.mode == 'rb':
            return _ffi_api.FileReadLineString(self)
        else:
            raise UnsupportedOperation('not readable')

    def has_nextline(self):
        return _ffi_api.FileHasNext(self)
