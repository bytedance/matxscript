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
# pylint: disable=redefined-builtin, wildcard-import

from ..._ffi import get_global_func as __get_global_func

__native_hash32 = __get_global_func("farmhash.hash32")
__native_hash64 = __get_global_func("farmhash.hash64")
__native_hash128 = __get_global_func("farmhash.hash128")

__native_hash32withseed = __get_global_func("farmhash.hash64withseed")
__native_hash64withseed = __get_global_func("farmhash.hash64withseed")
__native_hash128withseed = __get_global_func("farmhash.hash128withseed")

__native_fingerprint32 = __get_global_func("farmhash.fingerprint32")
__native_fingerprint64 = __get_global_func("farmhash.fingerprint64")
__native_fingerprint128 = __get_global_func("farmhash.fingerprint128")


def hash32(s: str):
    return __native_hash32(s)


def hash64(s: str):
    raise OverflowError("not supported for uint64, please use hash64_mod instead")
    native_h = __native_hash64(s)
    return native_h & 0xFFFFFFFFFFFFFFFF


def hash128(s: str):
    raise OverflowError("not supported for uint128")
    native_tup = __native_hash128(s)
    return native_tup[0] & 0xFFFFFFFFFFFFFFFF, native_tup[1] & 0xFFFFFFFFFFFFFFFF


def hash32withseed(s: str, seed: int):
    return __native_hash32withseed(s, seed)


def hash64withseed(s: str, seed: int):
    raise OverflowError("not supported for uint64, please use hash64withseed_mod instead")
    native_h = __native_hash64withseed(s, seed)
    return native_h & 0xFFFFFFFFFFFFFFFF


def hash128withseed(s: str, seedlow64: int, seedhigh64: int):
    raise OverflowError("not supported for uint128")
    native_tup = __native_hash32withseed(s, seedlow64, seedhigh64)
    return native_tup[0] & 0xFFFFFFFFFFFFFFFF, native_tup[1] & 0xFFFFFFFFFFFFFFFF


def fingerprint32(s: str):
    return __native_fingerprint32(s)


def fingerprint64(s: str):
    raise OverflowError("not supported for uint64, please use fingerprint64_mod instead")
    native_h = __native_fingerprint64(s)
    return native_h & 0xFFFFFFFFFFFFFFFF


def fingerprint128(s: str):
    raise OverflowError("not supported for uint128, please use fingerprint128_mod instead")
    native_tup = __native_fingerprint128(s)
    return native_tup[0] & 0xFFFFFFFFFFFFFFFF, native_tup[1] & 0xFFFFFFFFFFFFFFFF


###############################################################################
# for fix overflow, some sugar
###############################################################################

__native_hash64_mod = __get_global_func("farmhash.hash64_mod")
__native_hash64withseed_mod = __get_global_func("farmhash.hash64withseed_mod")

__native_fingerprint64_mod = __get_global_func("farmhash.fingerprint64_mod")
__native_fingerprint128_mod = __get_global_func("farmhash.fingerprint128_mod")


def hash64_mod(s: str, y: int):
    assert y <= 0x7FFFFFFFFFFFFFFF
    return __native_hash64_mod(s, y)


def hash64withseed_mod(s: str, seed: int, y: int):
    assert y <= 0x7FFFFFFFFFFFFFFF
    return __native_hash64withseed_mod(s, seed, y)


def fingerprint64_mod(s: str, y: int):
    assert y <= 0x7FFFFFFFFFFFFFFF
    return __native_fingerprint64_mod(s, y)


def fingerprint128_mod(s: str, y: int):
    assert y <= 0x7FFFFFFFFFFFFFFF
    return __native_fingerprint128_mod(s, y)
