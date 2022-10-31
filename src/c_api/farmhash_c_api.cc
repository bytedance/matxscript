// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <matxscript/runtime/pypi/kernel_farmhash.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash32").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.hash32] Expect 1 arguments but get " << args.size();
  return kernel_farmhash_hash32(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash64").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.hash64] Expect 1 arguments but get " << args.size();
  return kernel_farmhash_hash64(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash128").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.hash128] Expect 1 arguments but get " << args.size();
  return kernel_farmhash_hash128(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash32withseed").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[farmhash.hash32withseed] Expect 2 arguments but get "
                            << args.size();
  return kernel_farmhash_hash32withseed(args[0], args[1].As<uint32_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash64withseed").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[farmhash.hash64withseed] Expect 2 arguments but get "
                            << args.size();
  return kernel_farmhash_hash64withseed(args[0], args[1].As<uint64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash128withseed").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 3) << "[farmhash.hash128withseed] Expect 3 arguments but get "
                            << args.size();
  return kernel_farmhash_hash128withseed(args[0], args[1].As<uint64_t>(), args[2].As<uint64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.fingerprint32").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.fingerprint32] Expect 1 arguments but get "
                            << args.size();
  return kernel_farmhash_fingerprint32(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.fingerprint64").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.fingerprint64] Expect 1 arguments but get "
                            << args.size();
  return kernel_farmhash_fingerprint64(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.fingerprint128").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[farmhash.fingerprint128] Expect 1 arguments but get "
                            << args.size();
  return kernel_farmhash_fingerprint128(args[0]);
});

/******************************************************************************
 * for fix overflow, some sugar
 *****************************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash64_mod").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[farmhash.hash64_mod] Expect 2 arguments but get " << args.size();
  return kernel_farmhash_hash64_mod(args[0], args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.hash64withseed_mod").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 3) << "[farmhash.hash64withseed_mod] Expect 3 arguments but get "
                            << args.size();
  return kernel_farmhash_hash64withseed_mod(args[0], args[1].As<uint64_t>(), args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.fingerprint64_mod").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[farmhash.fingerprint64_mod] Expect 2 arguments but get "
                            << args.size();
  return kernel_farmhash_fingerprint64_mod(args[0], args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("farmhash.fingerprint128_mod").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[farmhash.fingerprint128_mod] Expect 2 arguments but get "
                            << args.size();
  return kernel_farmhash_fingerprint128_mod(args[0], args[1].As<int64_t>());
});

}  // namespace runtime
}  // namespace matxscript
