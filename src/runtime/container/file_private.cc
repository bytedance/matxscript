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
#include <matxscript/runtime/container/file_private.h>

#include <matxscript/runtime/container/file_ref.h>
#include <matxscript/runtime/container/list_ref.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * File container
 *****************************************************************************/
MATXSCRIPT_REGISTER_OBJECT_TYPE(FileNode);

/******************************************************************************
 * FileNode functions
 *****************************************************************************/

bool FileNode::HasNext() const {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  return !preader_->IsLastLine();
}

string_view FileNode::path() const {
  return path_;
}

String FileNode::ReadString(int64_t size) const {
  return preader_->Read(size);
}

static inline int UTF8DecodeOneLen(unsigned int c) {
  if (c < 0x80) { /* ascii? */
    return 1;
  } else {
    int count = 1;                /* to count number of continuation bytes */
    for (; c & 0x40u; c <<= 1u) { /* while it needs continuation bytes... */
      ++count;
    }
    return count;
  }
}

Unicode FileNode::ReadUnicode(int64_t size) const {
  String contents;
  int64_t count = size;
  int64_t skip_bytes = 0;
  while (count > 0) {
    int64_t remain_bytes = count > skip_bytes ? count : skip_bytes;
    String tmp = preader_->Read(remain_bytes);
    const char* data = tmp.data() + skip_bytes;
    int64_t limit = tmp.size() - skip_bytes;
    int64_t char_counts = 0;
    int64_t char_size = 0;
    while (data && limit > 0) {
      char_size = UTF8DecodeOneLen(data[0]);
      ++char_counts;
      if (char_size > limit) {
        break;
      }
      limit -= char_size;
      data += char_size;
    }
    if (limit > 0) {
      skip_bytes = char_size - limit;
    } else {
      skip_bytes = 0;
    }
    contents.append(tmp);
    count -= char_counts;
    if (preader_->IsLastLine()) {
      break;
    }
    if (count == 0 && skip_bytes > 0) {
      contents.append(preader_->Read(skip_bytes));
      break;
    }
  }
  return contents.decode();
}

RTValue FileNode::Read(int64_t size) const {
  if (binary_) {
    return {this->ReadString(size)};
  } else {
    return {this->ReadUnicode(size)};
  }
}

String FileNode::ReadLineString() const {
  // mode_ will not be checked, it's a simple file reader in c++
  // return empty String after reaching EOF, which is same in python
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  const char* line = nullptr;
  size_t len = 0;
  preader_->ReadLine(&line, &len);
  return String(line, len);
}

Unicode FileNode::ReadLineUnicode() const {
  // mode_ will not be checked, it's a simple file reader in c++
  // return empty Unicode after reaching EOF, which is same in python
  return ReadLineString().decode();
}

RTValue FileNode::Next() const {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  MXCHECK(readable_);
  if (binary_) {
    return RTValue(ReadLineString());
  } else {
    return RTValue(ReadLineUnicode());
  }
}

RTValue FileNode::Next(bool* has_next) const {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  MXCHECK(readable_);
  if (binary_) {
    RTValue ret(ReadLineString());
    *has_next = !preader_->IsLastLine();
    return ret;
  } else {
    RTValue ret(ReadLineUnicode());
    *has_next = !preader_->IsLastLine();
    return ret;
  }
}

RTView FileNode::NextView(bool* has_next, RTValue* holder_or_null) const {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  MXCHECK(readable_);
  if (binary_) {
    *holder_or_null = ReadLineString();
    *has_next = !preader_->IsLastLine();
    return *holder_or_null;
  } else {
    *holder_or_null = ReadLineUnicode();
    *has_next = !preader_->IsLastLine();
    return *holder_or_null;
  }
}

List FileNode::ReadLines() const {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  MXCHECK(readable_);
  List ret;
  const char* line = nullptr;
  size_t len = 0;
  while (preader_->ReadLine(&line, &len)) {
    if (binary_) {
      ret.append(String(line, len));
    } else {
      ret.append(String(line, len).decode());
    }
  }
  return ret;
}

void FileNode::Close() {
  MXCHECK(preader_ != nullptr) << "File is not opened!";
  preader_ = nullptr;
}

}  // namespace runtime
}  // namespace matxscript
