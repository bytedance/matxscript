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

#include <cstddef>
#include <iterator>

#include <matxscript/runtime/generator/generator_ref.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

#define DEFINE_GENERATOR_OBJECT_NODE(Name, ResultType)                             \
  class Name##Node : public Object {                                               \
   public:                                                                         \
    typedef GeneratorAdapter<ResultType> GAdapter;                                 \
                                                                                   \
   public:                                                                         \
    explicit Name##Node(GAdapter&& generator) : generator_(std::move(generator)) { \
    }                                                                              \
    explicit Name##Node() : generator_() {                                         \
    }                                                                              \
    ~Name##Node() = default;                                                       \
    static constexpr const uint32_t _type_index = TypeIndex::kRuntime##Name;       \
    static constexpr const char* _type_key = #Name;                                \
    MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(Name##Node, Object);                      \
                                                                                   \
   private:                                                                        \
    GAdapter generator_;                                                           \
    friend class Name;                                                             \
    friend class Name##NodeTrait;                                                  \
  }

DEFINE_GENERATOR_OBJECT_NODE(BoolGenerator, bool);
DEFINE_GENERATOR_OBJECT_NODE(Int32Generator, int32_t);
DEFINE_GENERATOR_OBJECT_NODE(Int64Generator, int64_t);
DEFINE_GENERATOR_OBJECT_NODE(Float32Generator, float);
DEFINE_GENERATOR_OBJECT_NODE(Float64Generator, double);
DEFINE_GENERATOR_OBJECT_NODE(RTValueGenerator, RTValue);

#undef DEFINE_GENERATOR_OBJECT_NODE
}  // namespace runtime
}  // namespace matxscript
