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


@_ffi.register_object("BoolGenerator")
class BoolGenerator(Object):

    def __iter__(self):
        return _ffi_api.BoolGenerator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)


@_ffi.register_object("Int32Generator")
class Int32Generator(Object):

    def __iter__(self):
        return _ffi_api.Int32Generator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)


@_ffi.register_object("Int64Generator")
class Int64Generator(Object):

    def __iter__(self):
        return _ffi_api.Int64Generator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)


@_ffi.register_object("Float32Generator")
class Float32Generator(Object):

    def __iter__(self):
        return _ffi_api.Float32Generator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)


@_ffi.register_object("Float64Generator")
class Float64Generator(Object):

    def __iter__(self):
        return _ffi_api.Float64Generator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)


@_ffi.register_object("RTValueGenerator")
class RTValueGenerator(Object):

    def __iter__(self):
        return _ffi_api.RTValueGenerator_Iter(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)
