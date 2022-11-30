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

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <string>

#include <matxscript/runtime/algorithm/prefix_mapping.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {
namespace extension {
namespace emoji {

class EmojiFilter {
 public:
  struct Options {
    bool unicode;              // general unicode emoji
    bool unicode_trans;        // general unicode emoji transcription
    bool unicode_trans_alias;  // general unicode emoji transcription alias
    RTValue user_codes;

    Options() : unicode(true), unicode_trans(false), unicode_trans_alias(false), user_codes() {
    }
  };

  explicit EmojiFilter(Options opt);
  virtual ~EmojiFilter() = default;

  // check ptr[pos: return_len] is emoji
  inline int CheckPos(const char* ptr, size_t len, int64_t pos = 0) const {
    if (pos) {

    }
    return emoji_codes_->PrefixSearch(ptr + pos, len, nullptr);
  }

  String Replace(const string_view& str, const string_view& repl, bool keep_all = true) const;

  String Filter(const string_view& str) const;

 private:
  Options opt_;
  std::shared_ptr<PrefixMapping> emoji_codes_;
};

}  // namespace emoji
}  // namespace extension
}  // namespace runtime
}  // namespace matxscript
