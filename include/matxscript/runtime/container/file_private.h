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

#include <algorithm>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/file_reader.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace runtime {

using PtrFileReader = std::shared_ptr<FileReader>;

/*! \brief file node content in file */
class FileNode : public Object {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeFile;
  static constexpr const char* _type_key = "runtime.File";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FileNode, Object);

  FileNode(const String& path, const String& mode = "r", const String& encoding = "utf-8")
      : path_(path), preader_(new FileReader(path, /* keep_newline */ true)) {
    // mode_ and _encoding are stored only for debug, they will not be checked while reading lines.
    mode_ = mode;
    std::transform(mode_.begin(), mode_.end(), mode_.begin(), ::tolower);
    MXCHECK(mode_ == "r" || mode_ == "rb") << "By now we only support \"r\" and \"rb\" mode.";
    encoding_ = encoding;
    std::transform(encoding_.begin(), encoding_.end(), encoding_.begin(), ::tolower);
    if (encoding_ == "utf8") {
      encoding_ = "utf-8";
    }
    MXCHECK(encoding_ == "utf-8") << "By now we only support \"utf-8\" encoding.";

    if (mode == "r") {
      readable_ = true;
    } else if (mode == "rb") {
      readable_ = true;
      binary_ = true;
    }
  }

  std::string GetRepr() const {
    std::ostringstream oss;
    oss << "File(\"" << path_ << "\", " << mode_ << ", " << encoding_ << ")";
    return oss.str();
  }

 public:
  bool HasNext() const;
  string_view path() const;
  String ReadString(int64_t size = -1) const;
  Unicode ReadUnicode(int64_t size = -1) const;
  RTValue Read(int64_t size = -1) const;
  String ReadLineString() const;
  Unicode ReadLineUnicode() const;
  List ReadLines() const;
  RTValue Next() const;
  RTValue Next(bool* has_next) const;
  RTView NextView(bool* has_next, RTValue* holder_or_null) const;
  void Close();

 private:
  // by now, we only support reader
  PtrFileReader preader_ = nullptr;
  String path_;
  String mode_;
  String encoding_;
  bool readable_ = false;
  bool binary_ = false;

  // Reference class
  friend class File;
  friend struct FileObjTrait;
};

}  // namespace runtime
}  // namespace matxscript
