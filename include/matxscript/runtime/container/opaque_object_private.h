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
#pragma once

#include <matxscript/runtime/container/opaque_object.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace runtime {

struct OpaqueObjectNode : public Object {
  void* ptr = nullptr;           // instance ptr
  int64_t tag = 0;               // for type check
  FOpaqueObjectDeleter deleter;  // for free ptr
  static constexpr int BUFFER_SIZE = 256;
  unsigned char buffer[BUFFER_SIZE];

  ~OpaqueObjectNode() {
    if (deleter && ptr) {
      (*deleter)(ptr);
    }
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeOpaqueObject;
  static constexpr const char* _type_key = "runtime.OpaqueObject";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(OpaqueObjectNode, Object);
};

}  // namespace runtime
}  // namespace matxscript
