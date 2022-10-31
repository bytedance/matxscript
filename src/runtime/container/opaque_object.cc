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
#include <matxscript/runtime/container/opaque_object.h>

#include <matxscript/runtime/container/opaque_object_private.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * OpaqueObject container
 *****************************************************************************/
OpaqueObject::OpaqueObject(int64_t tag, void* ptr, FOpaqueObjectDeleter deleter) {
  auto node = make_object<OpaqueObjectNode>();
  node->tag = tag;
  node->ptr = ptr;
  node->deleter = deleter;
  data_ = std::move(node);
}

void* OpaqueObject::GetOpaquePtr() const {
  MX_DPTR(OpaqueObject);
  return d ? d->ptr : nullptr;
}

int64_t OpaqueObject::GetTag() const {
  MX_DPTR(OpaqueObject);
  return d ? d->tag : -1;
}

int OpaqueObject::GetInternalBufferSize() {
  return OpaqueObjectNode::BUFFER_SIZE;
}

unsigned char* OpaqueObject::GetInternalBufferPtr() const {
  MX_DPTR(OpaqueObject);
  return d ? d->buffer : nullptr;
}

void OpaqueObject::update(int64_t tag, void* ptr, FOpaqueObjectDeleter deleter) const {
  MX_CHECK_DPTR(OpaqueObject);
  if (d->ptr != nullptr && d->deleter != nullptr && d->ptr != ptr) {
    d->deleter(d->ptr);
  }
  d->tag = tag;
  d->ptr = ptr;
  d->deleter = deleter;
}

std::ostream& operator<<(std::ostream& os, OpaqueObject const& n) {
  os << "OpaqueObject(code: " << n.GetTag() << ", ptr: " << n.GetOpaquePtr() << ")";
  return os;
}

}  // namespace runtime
}  // namespace matxscript
