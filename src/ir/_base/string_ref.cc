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
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/runtime/object_internal.h>

#include <initializer_list>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace ir {
// SEQualReduce traits for runtime containers.
struct StringNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const StringNode* key, SHashReducer hash_reduce) {
    hash_reduce->SHashReduceHashedValue(
        runtime::BytesHash(key->data_container.data(), key->data_container.size()));
  }

  static bool SEqualReduce(const StringNode* lhs, const StringNode* rhs, SEqualReducer equal) {
    if (lhs == rhs)
      return true;
    return lhs->data_container == rhs->data_container;
  }
};

StringNode::StringNode() {
}

StringNode::StringNode(DataContainer o) : data_container(std::move(o)) {
}

StringNode::operator StringNode::self_view() const noexcept {
  return data_container.view();
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(StringNode);
MATXSCRIPT_REGISTER_REFLECTION_VTABLE(StringNode, StringNodeTrait)
    .set_creator([](const runtime::String& bytes) {
      return runtime::ObjectInternal::GetObjectPtr(StringRef(bytes));
    })
    .set_repr_bytes([](const Object* n) -> runtime::String {
      return static_cast<const StringNode*>(n)->data_container;
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StringNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StringNode*>(node.get());
      p->stream << "b\'"
                << runtime::BytesEscape(op->data_container.data(), op->data_container.size())
                << "\'";
    });

/******************************************************************************
 * StringRef container
 *****************************************************************************/

const StringNode* StringRef::get() const {
  MX_DPTR(String);
  return d;
}

const StringRef::value_type* StringRef::data() const {
  MX_CHECK_DPTR(String);
  return d->data_container.data();
}

StringRef::operator StringRef::self_view() const noexcept {
  MX_DPTR(String);
  return d ? self_view{d->operator self_view()} : self_view();
}

StringRef::self_view StringRef::view() const noexcept {
  MX_DPTR(String);
  return d ? self_view{d->operator self_view()} : self_view();
}

StringRef::operator runtime::String() const {
  MX_DPTR(String);
  return d ? d->data_container : runtime::String{};
}

// default constructors
StringRef::StringRef() {
  data_ = runtime::make_object<StringNode>();
}

//  constructors from other
StringRef::StringRef(runtime::String other) {
  data_ = runtime::make_object<StringNode>(std::move(other));
}

StringRef::StringRef(const value_type* const data) : StringRef(runtime::String(data)) {
}

StringRef::StringRef(const value_type* data, size_type len)
    : StringRef(runtime::String(data, len)) {
}

// assign from other
StringRef& StringRef::operator=(runtime::String other) {
  MX_DPTR(String);
  if (d) {
    d->data_container = std::move(other);
  } else {
    data_ = runtime::make_object<StringNode>(std::move(other));
  }
  return *this;
}

StringRef& StringRef::operator=(const value_type* other) {
  return operator=(runtime::String(other));
}

// Member functions
StringNode* StringRef::CopyOnWrite() {
  if (data_.get() == nullptr) {
    data_ = runtime::make_object<StringNode>();
  } else if (!data_.unique()) {
    auto fbs = GetStringNode()->data_container;
    data_ = runtime::make_object<StringNode>(fbs);
  }
  return GetStringNode();
}

int StringRef::compare(const StringRef& other) const {
  return view().compare(other.view());
}

int StringRef::compare(const runtime::String& other) const {
  return view().compare(other);
}

int StringRef::compare(const char* other) const {
  return view().compare(other);
}

const char* StringRef::c_str() const {
  MX_CHECK_DPTR(String);
  return d->data_container.c_str();
}

int64_t StringRef::size() const {
  return view().size();
}

int64_t StringRef::length() const {
  return size();
}

bool StringRef::empty() const {
  return size() == 0;
}

StringRef StringRef::Concat(self_view lhs, self_view rhs) {
  StringNode::DataContainer container;
  container.reserve(lhs.size() + rhs.size());
  container.append(lhs.data(), lhs.size());
  container.append(rhs.data(), rhs.size());
  return StringRef(std::move(container));
}

// Private functions

StringNode* StringRef::GetStringNode() const {
  return static_cast<StringNode*>(data_.get());
}

StringNode* StringRef::CreateOrGetStringNode() {
  if (!data_.get()) {
    data_ = runtime::make_object<StringNode>();
  }
  return static_cast<StringNode*>(data_.get());
}

MATXSCRIPT_REGISTER_GLOBAL("runtime.String").set_body_typed([](runtime::String str) {
  return StringRef(std::move(str));
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetFFIString").set_body_typed([](StringRef str) {
  return str.operator runtime::String();
});

// runtime member function
MATXSCRIPT_REGISTER_GLOBAL("runtime.StringLen").set_body_typed([](StringRef str) {
  return static_cast<int64_t>(str.size());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.StringAdd").set_body_typed([](StringRef lhs, StringRef rhs) {
  return lhs + rhs;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.StringEqual").set_body_typed([](StringRef lhs, StringRef rhs) {
  return lhs == rhs;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.StringHash").set_body_typed([](StringRef str) {
  return static_cast<int64_t>(std::hash<StringRef>()(str));
});

/******************************************************************************
 * String iterators
 *****************************************************************************/

typename StringRef::iterator StringRef::begin() {
  auto n = CreateOrGetStringNode();
  return const_cast<StringRef::iterator>(n->data_container.data());
}

typename StringRef::const_iterator StringRef::begin() const {
  auto n = GetStringNode();
  MXCHECK(n != nullptr) << "[String.begin] container is null";
  return n->data_container.data();
}

typename StringRef::iterator StringRef::end() {
  auto n = CreateOrGetStringNode();
  return const_cast<StringRef::iterator>(n->data_container.data() + n->data_container.length());
}

typename StringRef::const_iterator StringRef::end() const {
  auto n = GetStringNode();
  MXCHECK(n != nullptr) << "[String.end] container is null";
  return n->data_container.data() + n->data_container.length();
}

typename StringRef::reverse_iterator StringRef::rbegin() {
  return reverse_iterator(end());
}

typename StringRef::const_reverse_iterator StringRef::rbegin() const {
  return const_reverse_iterator(end());
}

typename StringRef::reverse_iterator StringRef::rend() {
  return reverse_iterator(begin());
}

typename StringRef::const_reverse_iterator StringRef::rend() const {
  return const_reverse_iterator(begin());
}

}  // namespace ir
}  // namespace matxscript
