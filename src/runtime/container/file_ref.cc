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
#include <matxscript/runtime/container/file_ref.h>

#include <matxscript/runtime/container/file_private.h>
#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

template <>
bool IsConvertible<File>(const Object* node) {
  return node ? node->IsInstance<File::ContainerType>() : File::_type_is_nullable;
}

File::File(const Unicode& path, const Unicode& mode, const Unicode& encoding) {
  data_ = make_object<FileNode>(path.encode(), mode.encode(), encoding.encode());
}

File::File(File&& other) noexcept : ObjectRef() {  // NOLINT(*)
  data_ = std::move(other.data_);
}

File::File(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
}

File& File::operator=(File&& other) noexcept {
  data_ = std::move(other.data_);
  return *this;
}

const FileNode* File::operator->() const {
  return static_cast<const FileNode*>(data_.get());
}

const FileNode* File::get() const {
  return operator->();
}

bool File::HasNext() const {
  MX_CHECK_DPTR(File);
  return d->HasNext();
}

string_view File::path() const {
  MX_CHECK_DPTR(File);
  return d->path();
}

String File::ReadString(int64_t size) const {
  MX_CHECK_DPTR(File);
  return d->ReadString(size);
}

Unicode File::ReadUnicode(int64_t size) const {
  MX_CHECK_DPTR(File);
  return d->ReadUnicode(size);
}

String File::ReadLineString() const {
  MX_CHECK_DPTR(File);
  return d->ReadLineString();
}

RTValue File::Read(int64_t size) const {
  MX_CHECK_DPTR(File);
  return d->Read(size);
}

Unicode File::ReadLineUnicode() const {
  MX_CHECK_DPTR(File);
  return d->ReadLineUnicode();
}

List File::ReadLines() const {
  MX_CHECK_DPTR(File);
  return d->ReadLines();
}

RTValue File::Next() const {
  MX_CHECK_DPTR(File);
  return d->Next();
}

RTValue File::Next(bool* has_next) const {
  MX_CHECK_DPTR(File);
  return d->Next(has_next);
}

RTView File::NextView(bool* has_next, RTValue* holder_or_null) const {
  MX_CHECK_DPTR(File);
  return d->NextView(has_next, holder_or_null);
}

void File::close() const {
  MX_CHECK_DPTR(File);
  return d->Close();
}

std::ostream& operator<<(std::ostream& os, File const& n) {
  os << n->GetRepr();
  return os;
}

}  // namespace runtime
}  // namespace matxscript
