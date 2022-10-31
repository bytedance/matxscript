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
#include <matxscript/runtime/file_reader.h>

#include <string.h>
#include <stdexcept>

#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

size_t FileReader::_s_buf_size = 8 * 1024 * 1024;

FileReader::FileReader(string_view path, bool keep_newline) {
  _last_line = false;
  _keep_newline = keep_newline;
  _path = std::string(path.data(), path.size());
  _fd = open(_path.c_str(), O_RDONLY);
  MXCHECK_NE(_fd, -1) << "[FileReader] open file failed! \"" << _path.c_str()
                      << "\" maybe not exists.";
  _buffer.capacity = _s_buf_size;
  _buffer.data = new char[_buffer.capacity + 1];
  _buffer.position = 0;
  _buffer.limit = 0;
}

FileReader::~FileReader() {
  delete[] _buffer.data;
  if (_fd != -1) {
    close(_fd);
  }
}

String FileReader::Read(int64_t size) {
  String result;
  if (size < 0) {
    result.append(_buffer.data + _buffer.position, _buffer.limit - _buffer.position);
    _buffer.limit = 0;
    _buffer.position = 0;
    while (true) {
      ssize_t rs = read(_fd, _buffer.data, _buffer.capacity);
      if (rs <= 0) {
        _last_line = true;
        break;
      } else {
        result.append(_buffer.data, rs);
      }
    }
  } else {
    int64_t remaining = size;
    while (remaining > 0) {
      if (_buffer.limit - _buffer.position > remaining) {
        result.append(_buffer.data + _buffer.position, size);
        _buffer.position += size;
        break;
      } else {
        auto buf_remaining = _buffer.limit - _buffer.position;
        result.append(_buffer.data + _buffer.position, buf_remaining);
        remaining -= int64_t(buf_remaining);
        _buffer.limit = 0;
        _buffer.position = 0;
        if (_last_line) {
          // no more data
          break;
        }
        ssize_t rs = read(_fd, _buffer.data, _buffer.capacity);
        if (rs <= 0) {
          _last_line = true;
        } else {
          _buffer.limit = rs;
        }
      }
    }
  }
  return result;
}

bool FileReader::ReadLine(const char** line, size_t* len) {
  *len = 0;
  if (_last_line) {
    return false;
  }
  bool r = readLineFromBuffer(line, len);
  if (!r) {
    if (_buffer.position > 0) {
      if (_buffer.limit > _buffer.position) {
        memmove(_buffer.data, _buffer.data + _buffer.position, _buffer.limit - _buffer.position);
      }
      _buffer.limit = _buffer.limit - _buffer.position;
      _buffer.position = 0;
    }
    while (true) {
      ssize_t rs = read(_fd, _buffer.data + _buffer.limit, _buffer.capacity - _buffer.limit);
      if (rs <= 0) {
        _last_line = true;
        if (_buffer.limit > _buffer.position) {
          *line = _buffer.data + _buffer.position;
          *len = _buffer.limit - _buffer.position;
          r = true;
        }
        break;
      } else {
        _buffer.limit = _buffer.limit + rs;
        r = readLineFromBuffer(line, len);
        if (r) {
          break;
        }
      }
    }
  } else {
    // prefetch && set _last_line
    if (_buffer.position == _buffer.limit) {
      _buffer.limit = 0;
      _buffer.position = 0;
      if (_buffer.capacity == *len) {
        // large line need more buffer
        size_t capacity = 2 * _buffer.capacity;
        char* nb_data = new char[capacity + 1];
        memset(nb_data, 0, capacity + 1);
        memcpy(nb_data, _buffer.data, _buffer.capacity);
        delete[] _buffer.data;
        _buffer.data = nb_data;
        _buffer.capacity = capacity;
      }
      ssize_t rs = read(_fd, _buffer.data, _buffer.capacity - *len);
      if (rs <= 0) {
        _last_line = true;
      } else {
        _buffer.limit = rs;
      }
    }
  }
  return r;
}

bool FileReader::readLineFromBuffer(const char** line, size_t* len) {
  if (_buffer.position == _buffer.limit) {
    return false;
  }
  size_t start = _buffer.position;
  size_t end = start;
  while (end < _buffer.limit && _buffer.data[end] != '\n') {
    ++end;
  }
  if (end == _buffer.limit) {
    if (start == 0 && end + 1 >= _buffer.capacity) {
      size_t capacity = 2 * _buffer.capacity;
      char* nb_data = new char[capacity + 1];
      memset(nb_data, 0, capacity + 1);
      memcpy(nb_data, _buffer.data, _buffer.capacity);
      delete[] _buffer.data;
      _buffer.data = nb_data;
      _buffer.capacity = capacity;
    }
    return false;
  }
  *line = _buffer.data + start;
  if (!_keep_newline) {
    if (end > start && _buffer.data[end - 1] == '\r') {
      *len = end - start - 1;
    } else {
      *len = end - start;
    }
  } else {
    *len = end - start + 1;
  }
  _buffer.position = end + 1;
  return true;
}

}  // namespace runtime
}  // namespace matxscript
