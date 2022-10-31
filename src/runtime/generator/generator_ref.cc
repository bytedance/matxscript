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
#include <matxscript/runtime/generator/generator_ref.h>

#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/generator/generator_private.h>
#include <matxscript/runtime/memory.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Generator
 *****************************************************************************/

#define DEFINE_GENERATOR_OBJECT_REF(Name, ResultType)                                \
  Name::Name(std::shared_ptr<BaseGenerator<ResultType>> generator) {                 \
    data_ = make_object<Name##Node>(Name::GAdapter(std::move(generator)));           \
  }                                                                                  \
  Name::Name(GAdapter generator) {                                                   \
    data_ = make_object<Name##Node>(std::move(generator));                           \
  }                                                                                  \
  Name::GAdapter* Name::GetGenerator() const {                                       \
    MX_CHECK_DPTR(Name);                                                             \
    return &d->generator_;                                                           \
  }                                                                                  \
  Iterator Name::iter() const {                                                      \
    return MakeGenericIterator(*this, begin(), end());                               \
  }                                                                                  \
  template <>                                                                        \
  bool IsConvertible<Name>(const Object* node) {                                     \
    return node ? node->IsInstance<Name::ContainerType>() : Name::_type_is_nullable; \
  }                                                                                  \
  std::ostream& operator<<(std::ostream& os, Name const& n) {                        \
    os << #Name;                                                                     \
    return os;                                                                       \
  }

DEFINE_GENERATOR_OBJECT_REF(BoolGenerator, bool);
DEFINE_GENERATOR_OBJECT_REF(Int32Generator, int32_t);
DEFINE_GENERATOR_OBJECT_REF(Int64Generator, int64_t);
DEFINE_GENERATOR_OBJECT_REF(Float32Generator, float);
DEFINE_GENERATOR_OBJECT_REF(Float64Generator, double);
DEFINE_GENERATOR_OBJECT_REF(RTValueGenerator, RTValue);

}  // namespace runtime
}  // namespace matxscript
