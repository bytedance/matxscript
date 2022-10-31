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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/generator/generator_ref.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.BoolGenerator_Iter").set_body([](PyArgs args) -> RTValue {
  BoolGenerator container = args[0].As<BoolGenerator>();
  return container.iter();
});
MATXSCRIPT_REGISTER_GLOBAL("runtime.Int32Generator_Iter").set_body([](PyArgs args) -> RTValue {
  Int32Generator container = args[0].As<Int32Generator>();
  return container.iter();
});
MATXSCRIPT_REGISTER_GLOBAL("runtime.Int64Generator_Iter").set_body([](PyArgs args) -> RTValue {
  Int64Generator container = args[0].As<Int64Generator>();
  return container.iter();
});
MATXSCRIPT_REGISTER_GLOBAL("runtime.Float32Generator_Iter").set_body([](PyArgs args) -> RTValue {
  Float32Generator container = args[0].As<Float32Generator>();
  return container.iter();
});
MATXSCRIPT_REGISTER_GLOBAL("runtime.Float64Generator_Iter").set_body([](PyArgs args) -> RTValue {
  Float64Generator container = args[0].As<Float64Generator>();
  return container.iter();
});
MATXSCRIPT_REGISTER_GLOBAL("runtime.RTValueGenerator_Iter").set_body([](PyArgs args) -> RTValue {
  RTValueGenerator container = args[0].As<RTValueGenerator>();
  return container.iter();
});

}  // namespace runtime
}  // namespace matxscript
