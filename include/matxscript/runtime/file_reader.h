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

#include <fcntl.h>     /* for open */
#include <sys/stat.h>  /* for open */
#include <sys/types.h> /* for open */
#include <unistd.h>    /* for lseek and write */

#include <string>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>

namespace matxscript {
namespace runtime {

class FileReader {
  struct ByteBuffer {
    char* data = nullptr;
    size_t capacity = 0;
    size_t position = 0;
    size_t limit = 0;
  };

 public:
  explicit FileReader(string_view path, bool keep_newline = false);

  virtual ~FileReader();

 public:
  bool ReadLine(std::string& line) {
    const char* out = nullptr;
    size_t len = 0;
    bool r = ReadLine(&out, &len);
    line = std::move(std::string(out, len));
    return r;
  }

  bool ReadLine(const char** line, size_t* len);

  inline bool IsLastLine() const {
    return _last_line;
  }

  String Read(int64_t size);

 private:
  bool readLineFromBuffer(const char** line, size_t* len);

 private:
  int _fd;
  ByteBuffer _buffer;
  std::string _path;
  bool _last_line;
  static size_t _s_buf_size;
  bool _keep_newline = false;
};

}  // namespace runtime
}  // namespace matxscript
