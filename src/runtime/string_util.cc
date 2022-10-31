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
#include <matxscript/runtime/string_util.h>

#include <iconv.h>

#include <cstdarg>
#include <cstring>
#include <locale>
#include <stdexcept>

namespace matxscript {
namespace runtime {
namespace StringUtil {

std::string Format(const char* fmt, ...) {
  char msg[40960] = {0};
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);
  return std::move(std::string(msg));
}

std::string& Replace(std::string& input, const std::string& t, const std::string& r) {
  std::string::size_type pos = 0;
  std::string::size_type t_len = t.size();
  std::string::size_type r_len = r.size();
  while ((pos = input.find(t, pos)) != std::string::npos) {
    input.replace(pos, t_len, r);
    pos += r_len;
  }
  return input;
}

std::vector<std::string> Split(const std::string& input,
                               const std::string& seps,
                               bool preserve_all_tokens) {
  int max = -1;
  std::vector<std::string> result;
  if (input.empty()) {
    return std::move(result);
  }
  unsigned int len = input.length();
  unsigned int size_plus1 = 1;
  unsigned int i = 0;
  unsigned int start = 0;
  bool match = false;
  bool last_match = false;
  if (seps.length() == 1) {
    // Optimise 1 character case
    char sep = seps.at(0);
    while (i < len) {
      if (input.at(i) == sep) {
        if (match || preserve_all_tokens) {
          last_match = true;
          if (size_plus1++ == max) {
            i = len;
            last_match = false;
          }
          result.push_back(input.substr(start, i - start));
          match = false;
        }
        start = ++i;
        continue;
      }
      last_match = false;
      match = true;
      i++;
    }
  } else {
    // standard case
    while (i < len) {
      if (seps.find(input.at(i)) != std::string::npos) {
        if (match || preserve_all_tokens) {
          last_match = true;
          if (size_plus1++ == max) {
            i = len;
            last_match = false;
          }
          result.push_back(input.substr(start, i - start));
          match = false;
        }
        start = ++i;
        continue;
      }
      last_match = false;
      match = true;
      i++;
    }
  }
  if (match || (preserve_all_tokens && last_match)) {
    result.push_back(input.substr(start, i));
  }
  return std::move(result);
}

std::string Concat(const std::vector<std::string>& inputs, const std::string& sep) {
  std::string result;
  size_t len = 0;
  for (auto& w : inputs) {
    len += w.size() + sep.size();
  }
  result.reserve(len);
  auto begin = inputs.begin();
  auto end = inputs.end();
  if (begin != end) {
    result.append(*begin);
    ++begin;
  }
  for (; begin != end; ++begin) {
    result.append(sep);
    result.append(*begin);
  }
  return std::move(result);
}

bool Find(const std::string& source, const std::string& target) {
  int i = Find(source.c_str(), source.size(), target.c_str(), target.size());
  return i >= 0;
}

int Find(const char* source, size_t source_len, const char* target, size_t target_len) {
  int i = 0;
  int j = 0;
  if (!source || source[0] == '\0') {
    return -1;
  }
  if (!target || target[0] == '\0') {
    return 0;
  }
  if (source_len <= 0) {
    source_len = strlen(source);
  }
  if (target_len <= 0) {
    target_len = strlen(target);
  }
  if (source_len < target_len) {
    return -1;
  }
  while (i < source_len) {
    if (source[i] == target[j]) {
      if (j == target_len - 1) {
        return i - target_len + 1;
      }
      ++i;
      ++j;
    } else {
      int index = -1;
      int t = i - j + target_len;
      for (j = target_len - 1; j >= 0; --j) {
        if (target[j] == source[t]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        i = t + 1;
      } else {
        i = t - index;
      }
      j = 0;
    }
  }
  return -1;
}

}  // namespace StringUtil
}  // namespace runtime
}  // namespace matxscript
