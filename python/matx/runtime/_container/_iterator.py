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
from ..object_generic import to_runtime_object


@_ffi.register_object("Iterator")
class Iterator(Object):

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __next__(self):
        if _ffi_api.Iterator_HasNext(self):
            return _ffi_api.Iterator_Next(self)
        else:
            raise StopIteration()

    def __iter__(self):
        return self
