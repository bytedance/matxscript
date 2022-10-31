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

from ... import _ffi
from .. import _ffi_api
from ..object import Object


@_ffi.register_object("runtime.UserData")
class UserData(Object):

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.UserData)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __getattr__(self, name):
        return _ffi_api.UserData___getattr__(self, name.encode())

    def __call__(self, *args):
        return _ffi_api.UserData___call__(self, *args)
