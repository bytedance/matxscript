// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
 *
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

#include <matxscript/ir/_base/object_path.h>

#include <algorithm>
#include <cstring>

#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/registry.h>
#include "matxscript/ir/_base/reflection.h"

using namespace matxscript::runtime;

namespace matxscript {
namespace ir {

// ============== ObjectPathNode ==============

ObjectPathNode::ObjectPathNode(const ObjectPathNode* parent)
    : parent_(GetRef<ObjectRef>(parent)), length_(parent == nullptr ? 1 : parent->length_ + 1) {
}

// --- GetParent ---

Optional<ObjectPath> ObjectPathNode::GetParent() const {
  if (parent_ == nullptr) {
    return NullOpt;
  } else {
    return Downcast<ObjectPath>(parent_);
  }
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathGetParent").set_body_typed([](const ObjectPath& self) {
  return self->GetParent();
});

// --- Length ---

int32_t ObjectPathNode::Length() const {
  return length_;
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathLength").set_body_typed([](const ObjectPath& self) {
  return self->Length();
});

// --- GetPrefix ---

ObjectPath ObjectPathNode::GetPrefix(int32_t length) const {
  MXCHECK_GE(length, 1) << "IndexError: Prefix length must be at least 1";
  MXCHECK_LE(length, Length())
      << "IndexError: Attempted to get a prefix longer than the path itself";

  const ObjectPathNode* node = this;
  int32_t suffix_len = Length() - length;
  for (int32_t i = 0; i < suffix_len; ++i) {
    node = node->ParentNode();
  }

  return GetRef<ObjectPath>(node);
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathGetPrefix")
    .set_body_typed([](const ObjectPath& self, int32_t length) { return self->GetPrefix(length); });

// --- IsPrefixOf ---

bool ObjectPathNode::IsPrefixOf(const ObjectPath& other) const {
  int32_t this_len = Length();
  if (this_len > other->Length()) {
    return false;
  }
  return this->PathsEqual(other->GetPrefix(this_len));
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathIsPrefixOf")
    .set_body_typed([](const ObjectPath& self, const ObjectPath& other) {
      return self->IsPrefixOf(other);
    });

// --- Attr ---

ObjectPath ObjectPathNode::Attr(const char* attr_key) const {
  if (attr_key != nullptr) {
    return ObjectPath(make_object<AttributeAccessPathNode>(this, attr_key));
  } else {
    return ObjectPath(make_object<UnknownAttributeAccessPathNode>(this));
  }
}

ObjectPath ObjectPathNode::Attr(Optional<StringRef> attr_key) const {
  if (attr_key.defined()) {
    return ObjectPath(make_object<AttributeAccessPathNode>(this, attr_key.value()));
  } else {
    return ObjectPath(make_object<UnknownAttributeAccessPathNode>(this));
  }
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathAttr")
    .set_body_typed([](const ObjectPath& object_path, const Optional<StringRef>& attr_key) {
      return object_path->Attr(attr_key);
    });

// --- ArrayIndex ---

ObjectPath ObjectPathNode::ArrayIndex(int32_t index) const {
  return ObjectPath(make_object<ArrayIndexPathNode>(this, index));
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathArrayIndex")
    .set_body_typed([](const ObjectPath& self, int32_t index) { return self->ArrayIndex(index); });

// --- MissingArrayElement ---

ObjectPath ObjectPathNode::MissingArrayElement(int32_t index) const {
  return ObjectPath(make_object<MissingArrayElementPathNode>(this, index));
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathMissingArrayElement")
    .set_body_typed([](const ObjectPath& self, int32_t index) {
      return self->MissingArrayElement(index);
    });

// --- MapValue ---

ObjectPath ObjectPathNode::MapValue(ObjectRef key) const {
  return ObjectPath(make_object<MapValuePathNode>(this, std::move(key)));
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathMapValue")
    .set_body_typed([](const ObjectPath& self, const ObjectRef& key) {
      return self->MapValue(key);
    });

// --- MissingMapEntry ---

ObjectPath ObjectPathNode::MissingMapEntry() const {
  return ObjectPath(make_object<MissingMapEntryPathNode>(this));
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathMissingMapEntry")
    .set_body_typed([](const ObjectPath& self) { return self->MissingMapEntry(); });

// --- PathsEqual ----

bool ObjectPathNode::PathsEqual(const ObjectPath& other) const {
  if (!other.defined() || Length() != other->Length()) {
    return false;
  }

  const ObjectPathNode* lhs = this;
  const ObjectPathNode* rhs = static_cast<const ObjectPathNode*>(other.get());

  while (lhs != nullptr && rhs != nullptr) {
    if (lhs->type_index() != rhs->type_index()) {
      return false;
    }
    if (!lhs->LastNodeEqual(rhs)) {
      return false;
    }
    lhs = lhs->ParentNode();
    rhs = rhs->ParentNode();
  }

  return lhs == nullptr && rhs == nullptr;
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathEqual")
    .set_body_typed([](const ObjectPath& self, const ObjectPath& other) {
      return self->PathsEqual(other);
    });

// --- Repr ---

runtime::String GetObjectPathRepr(const ObjectPathNode* node) {
  runtime::String ret;
  while (node != nullptr) {
    runtime::String node_str = node->LastNodeString();
    ret.append(node_str.rbegin(), node_str.rend());
    node = static_cast<const ObjectPathNode*>(node->GetParent().get());
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

static void PrintObjectPathRepr(const ObjectRef& node, ReprPrinter* p) {
  p->stream << GetObjectPathRepr(static_cast<const ObjectPathNode*>(node.get()));
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(ObjectPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<ObjectPathNode>(PrintObjectPathRepr);

// --- Private/protected methods ---

const ObjectPathNode* ObjectPathNode::ParentNode() const {
  return static_cast<const ObjectPathNode*>(parent_.get());
}

// ============== ObjectPath ==============

/* static */ ObjectPath ObjectPath::Root() {
  return ObjectPath(make_object<RootPathNode>());
}

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathRoot").set_body_typed(ObjectPath::Root);

// ============== Individual path classes ==============

// ----- Root -----

RootPathNode::RootPathNode() : ObjectPathNode(nullptr) {
}

bool RootPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  return true;
}

runtime::String RootPathNode::LastNodeString() const {
  return "<root>";
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(RootPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<RootPathNode>(PrintObjectPathRepr);

// ----- AttributeAccess -----

AttributeAccessPathNode::AttributeAccessPathNode(const ObjectPathNode* parent, StringRef attr_key)
    : ObjectPathNode(parent), attr_key(std::move(attr_key)) {
}

bool AttributeAccessPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherAttrAccess = static_cast<const AttributeAccessPathNode*>(other);
  return attr_key == otherAttrAccess->attr_key;
}

runtime::String AttributeAccessPathNode::LastNodeString() const {
  return "." + attr_key;
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(AttributeAccessPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttributeAccessPathNode>(PrintObjectPathRepr);

// ----- UnknownAttributeAccess -----

UnknownAttributeAccessPathNode::UnknownAttributeAccessPathNode(const ObjectPathNode* parent)
    : ObjectPathNode(parent) {
}

bool UnknownAttributeAccessPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  // Consider any two unknown attribute accesses unequal
  return false;
}

runtime::String UnknownAttributeAccessPathNode::LastNodeString() const {
  return ".<unknown attribute>";
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(UnknownAttributeAccessPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnknownAttributeAccessPathNode>(PrintObjectPathRepr);

// ----- ArrayIndexPath -----

ArrayIndexPathNode::ArrayIndexPathNode(const ObjectPathNode* parent, int32_t index)
    : ObjectPathNode(parent), index(index) {
}

bool ArrayIndexPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherArrayIndex = static_cast<const ArrayIndexPathNode*>(other);
  return index == otherArrayIndex->index;
}

runtime::String ArrayIndexPathNode::LastNodeString() const {
  return "[" + std::to_string(index) + "]";
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(ArrayIndexPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ArrayIndexPathNode>(PrintObjectPathRepr);

// ----- MissingArrayElement -----

MissingArrayElementPathNode::MissingArrayElementPathNode(const ObjectPathNode* parent,
                                                         int32_t index)
    : ObjectPathNode(parent), index(index) {
}

bool MissingArrayElementPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherMissingElement = static_cast<const MissingArrayElementPathNode*>(other);
  return index == otherMissingElement->index;
}

runtime::String MissingArrayElementPathNode::LastNodeString() const {
  return "[<missing element #" + std::to_string(index) + ">]";
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(MissingArrayElementPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MissingArrayElementPathNode>(PrintObjectPathRepr);

// ----- MapValue -----

MapValuePathNode::MapValuePathNode(const ObjectPathNode* parent, ObjectRef key)
    : ObjectPathNode(parent), key(std::move(key)) {
}

bool MapValuePathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherMapValue = static_cast<const MapValuePathNode*>(other);
  return ObjectEqual()(key, otherMapValue->key);
}

runtime::String MapValuePathNode::LastNodeString() const {
  std::ostringstream s;
  s << "[" << key << "]";
  return s.str();
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(MapValuePathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MapValuePathNode>(PrintObjectPathRepr);

// ----- MissingMapEntry -----

MissingMapEntryPathNode::MissingMapEntryPathNode(const ObjectPathNode* parent)
    : ObjectPathNode(parent) {
}

bool MissingMapEntryPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  return true;
}

runtime::String MissingMapEntryPathNode::LastNodeString() const {
  return "[<missing entry>]";
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(MissingMapEntryPathNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MissingMapEntryPathNode>(PrintObjectPathRepr);

MATXSCRIPT_REGISTER_OBJECT_TYPE(ObjectPathPairNode);

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathPairLhsPath")
    .set_body_typed([](const ObjectPathPair& object_path_pair) {
      return object_path_pair->lhs_path;
    });

MATXSCRIPT_REGISTER_GLOBAL("node.ObjectPathPairRhsPath")
    .set_body_typed([](const ObjectPathPair& object_path_pair) {
      return object_path_pair->rhs_path;
    });

ObjectPathPairNode::ObjectPathPairNode(ObjectPath lhs_path, ObjectPath rhs_path)
    : lhs_path(std::move(lhs_path)), rhs_path(std::move(rhs_path)) {
}

ObjectPathPair::ObjectPathPair(ObjectPath lhs_path, ObjectPath rhs_path) {
  data_ = make_object<ObjectPathPairNode>(std::move(lhs_path), std::move(rhs_path));
}

}  // namespace ir
}  // namespace matxscript
