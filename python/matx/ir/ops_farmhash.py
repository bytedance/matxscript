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
# pylint: disable=redefined-builtin, invalid-name

from .op import hlo_call_intrin
from . import type as _type

_self_module_name = "matx.pypi.farmhash"


###############################################################################
# origin functions
###############################################################################

def farmhash_hash32(span, s):
    func_name = 'ir.farmhash_hash32'
    return hlo_call_intrin(_type.PrimType("uint32"), func_name, span, s)


def farmhash_hash64(span, s):
    func_name = 'ir.farmhash_hash64'
    return hlo_call_intrin(_type.PrimType("uint64"), func_name, span, s)


def farmhash_hash128(span, s):
    func_name = 'ir.farmhash_hash32'
    ret_type = _type.TupleType([_type.PrimType("uint64"), _type.PrimType("uint64")])
    return hlo_call_intrin(ret_type, func_name, span, s)


def farmhash_hash32withseed(span, s, seed):
    func_name = 'ir.farmhash_hash32withseed'
    return hlo_call_intrin(_type.PrimType("uint32"), func_name, span, s, seed)


def farmhash_hash64withseed(span, s, seed):
    func_name = 'ir.farmhash_hash64withseed'
    return hlo_call_intrin(_type.PrimType("uint64"), func_name, span, s, seed)


def farmhash_hash128withseed(span, s, seedlow64, seedhigh64):
    func_name = 'ir.farmhash_hash128withseed'
    ret_type = _type.TupleType([_type.PrimType("uint64"), _type.PrimType("uint64")])
    return hlo_call_intrin(ret_type, func_name, span, s, seedlow64, seedhigh64)


def farmhash_fingerprint32(span, s):
    func_name = 'ir.farmhash_fingerprint32'
    return hlo_call_intrin(_type.PrimType("uint32"), func_name, span, s)


def farmhash_fingerprint64(span, s):
    func_name = 'ir.farmhash_fingerprint64'
    return hlo_call_intrin(_type.PrimType("uint64"), func_name, span, s)


def farmhash_fingerprint128(span, s):
    func_name = 'ir.farmhash_fingerprint128'
    ret_type = _type.TupleType([_type.PrimType("uint64"), _type.PrimType("uint64")])
    return hlo_call_intrin(ret_type, func_name, span, s)


###############################################################################
# for fix overflow, some sugar
###############################################################################

def farmhash_hash64_mod(span, s, y):
    func_name = 'ir.farmhash_hash64_mod'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, s, y)


def farmhash_hash64withseed_mod(span, s, seed, y):
    func_name = 'ir.farmhash_hash64withseed_mod'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, s, seed, y)


def farmhash_fingerprint64_mod(span, s, y):
    func_name = 'ir.farmhash_fingerprint64_mod'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, s, y)


def farmhash_fingerprint128_mod(span, s, y):
    func_name = 'ir.farmhash_fingerprint128_mod'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, s, y)


###############################################################################
# registry info for parser
###############################################################################

registry_info = {
    f'{_self_module_name}.hash32': farmhash_hash32,
    f'{_self_module_name}.hash64': farmhash_hash64,
    f'{_self_module_name}.hash128': farmhash_hash128,
    f'{_self_module_name}.hash32withseed': farmhash_hash32withseed,
    f'{_self_module_name}.hash64withseed': farmhash_hash64withseed,
    f'{_self_module_name}.hash128withseed': farmhash_hash128withseed,
    f'{_self_module_name}.fingerprint32': farmhash_fingerprint32,
    f'{_self_module_name}.fingerprint64': farmhash_fingerprint64,
    f'{_self_module_name}.fingerprint128': farmhash_fingerprint128,

    f'{_self_module_name}.hash64_mod': farmhash_hash64_mod,
    f'{_self_module_name}.hash64withseed_mod': farmhash_hash64withseed_mod,
    f'{_self_module_name}.fingerprint64_mod': farmhash_fingerprint64_mod,
    f'{_self_module_name}.fingerprint128_mod': farmhash_fingerprint128_mod,
}
