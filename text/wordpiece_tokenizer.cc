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
#include <map>

#include <matxscript/runtime/algorithm/cedar.h>
#include <matxscript/runtime/algorithm/prefix_mapping.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/file_reader.h>
#include <matxscript/runtime/file_util.h>
#include "matxscript/runtime/container/string_helper.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/type_helper_macros.h"

#include "common_funcs.h"

namespace matxscript {
namespace runtime {
namespace extension {
namespace tokenizer {

class WordPieceTokenizer {
 public:
  WordPieceTokenizer(String vocab_path,
                     bool lookup_id,
                     const Any& unk_token,
                     String subwords_prefix,
                     bool skip_empty,
                     int max_bytes_per_token);
  virtual ~WordPieceTokenizer() = default;

 public:
  void tokenize(const string_view& raw_token, const List& output_tokens) const;
  void tokenize(const unicode_view& raw_token, const List& output_tokens) const;
  void tokenize(const List& sentence, const List& output_tokens) const;
  void tokenize(const List& sentence, const List& output_tokens, const List& output_lens) const;

 public:
  RTValue tokenize(PyArgs args);
  RTValue tokenize_with_meta(PyArgs args);

 private:
  template <class PostFunction>
  inline void TokenizeImplWithPrefix(const char* token_buf,
                                     int64_t token_len,
                                     const PostFunction& post_func,
                                     const List& output_tokens) const;

  template <class PostFunction>
  inline void TokenizeImplNoPrefix(const char* token_buf,
                                   int64_t token_len,
                                   const PostFunction& post_func,
                                   const List& output_tokens) const;

 private:
  String vocab_path_;
  bool skip_empty_;
  bool lookup_id_;
  int max_bytes_per_token_;
  String unk_token_;
  int unk_id_;
  String subwords_prefix_;
  std::shared_ptr<PrefixMapping> prefix_matcher_;
};

WordPieceTokenizer::WordPieceTokenizer(String vocab_path,
                                       bool lookup_id,
                                       const Any& unk_token,
                                       String subwords_prefix,
                                       bool skip_empty,
                                       int max_bytes_per_token) {
  if (unk_token.is_nullptr()) {
    MXCHECK(!lookup_id) << "unk_token must not be None when lookup_id is True";
  }
  unk_token_ = commons::details::GetString(unk_token, __FILE__, __LINE__);
  subwords_prefix_ = std::move(subwords_prefix);
  skip_empty_ = skip_empty;
  lookup_id_ = lookup_id;
  max_bytes_per_token_ = max_bytes_per_token;
  vocab_path_ = std::move(vocab_path);
  MXCHECK(FileUtil::Exists(vocab_path_)) << "vocab file \"" << vocab_path_ << "\" not exists!";

  std::map<String, int> tokens;
  FileReader reader(vocab_path_);
  const char* line = nullptr;
  size_t line_len = 0;
  int64_t line_no = 0;
  while (reader.ReadLine(&line, &line_len)) {
    // Ignoring empty lines
    if (line_len == 0) {
      continue;
    }
    tokens.emplace(String(line, line_len), line_no);
    ++line_no;
  }
  prefix_matcher_ = std::make_shared<PrefixMapping>(tokens);
  if (unk_token.is_nullptr()) {
    unk_id_ = -1;
  } else {
    auto match_len = prefix_matcher_->PrefixSearch(unk_token_.data(), unk_token_.size(), &unk_id_);
    MXCHECK(match_len == unk_token_.size()) << "unk_token \'" << unk_token_ << "\' not found";
  }
}

template <class PostFunction>
inline void WordPieceTokenizer::TokenizeImplWithPrefix(const char* token_buf,
                                                       int64_t token_len,
                                                       const PostFunction& post_func,
                                                       const List& output_tokens) const {
  commons::details::SmallBuffer<512> small_subword_buffer(token_len + subwords_prefix_.size());
  char* subword_buf = small_subword_buffer.Data();

  auto* sub_prefix_ptr = subwords_prefix_.data();
  auto sub_prefix_size = subwords_prefix_.size();
  std::memcpy(subword_buf, sub_prefix_ptr, sub_prefix_size);

  if (skip_empty_ && token_len == 0) {
    // strip empty word
    return;
  }
  if (token_len > max_bytes_per_token_) {
    post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
    return;
  }
  auto token_ptr = token_buf;
  int value = -1;
  int match_len = prefix_matcher_->PrefixSearch(token_ptr, token_len, &value);
  if (match_len == token_len) {
    // full match
    post_func(token_ptr, match_len, value, output_tokens);
  } else if (match_len == 0) {
    // zero match
    post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
  } else {
    // partial match
    int count = 1;
    post_func(token_ptr, match_len, value, output_tokens);
    token_len -= match_len;
    token_ptr += match_len;
    int subword_len = token_len;
    char* subword_ptr = subword_buf;
    std::memcpy(subword_ptr + sub_prefix_size, token_ptr, token_len);
    subword_len += sub_prefix_size;

    while (subword_len > 0) {
      match_len = prefix_matcher_->PrefixSearch(subword_ptr, subword_len, &value);
      if (match_len == subword_len) {
        post_func(subword_ptr, match_len, value, output_tokens);
        break;
      } else if (match_len <= sub_prefix_size) {
        for (int bi = 0; bi < count; ++bi) {
          output_tokens.pop_back();
        }
        post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
        break;
      } else {
        post_func(subword_ptr, match_len, value, output_tokens);
        subword_len -= match_len - sub_prefix_size;
        subword_ptr += match_len - sub_prefix_size;
        std::memcpy(subword_ptr, sub_prefix_ptr, sub_prefix_size);
        ++count;
      }
    }
  }
}

template <class PostFunction>
inline void WordPieceTokenizer::TokenizeImplNoPrefix(const char* token_buf,
                                                     int64_t token_len,
                                                     const PostFunction& post_func,
                                                     const List& output_tokens) const {
  if (skip_empty_ && token_len == 0) {
    // strip empty word
    return;
  }
  if (token_len > max_bytes_per_token_) {
    post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
    return;
  }
  auto token_ptr = token_buf;
  int value = -1;
  int match_len = prefix_matcher_->PrefixSearch(token_ptr, token_len, &value);
  if (match_len == token_len) {
    // full match
    post_func(token_ptr, match_len, value, output_tokens);
  } else if (match_len == 0) {
    // zero match
    post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
  } else {
    // partial match
    int count = 1;
    post_func(token_ptr, match_len, value, output_tokens);
    token_len -= match_len;
    token_ptr += match_len;
    while (token_len > 0) {
      match_len = prefix_matcher_->PrefixSearch(token_ptr, token_len, &value);
      if (match_len == token_len) {
        post_func(token_ptr, match_len, value, output_tokens);
        break;
      } else if (match_len == 0) {
        for (int bi = 0; bi < count; ++bi) {
          output_tokens.pop_back();
        }
        post_func(unk_token_.data(), unk_token_.size(), unk_id_, output_tokens);
        break;
      } else {
        post_func(token_ptr, match_len, value, output_tokens);
        token_len -= match_len;
        token_ptr += match_len;
        ++count;
      }
    }
  }
}

void WordPieceTokenizer::tokenize(const string_view& raw_token, const List& output_tokens) const {
  if (lookup_id_) {
    auto post_func = [](const char* token_buf, int token_len, int value, const List& output) {
      output.push_back(value);
    };
    if (subwords_prefix_.empty()) {
      TokenizeImplNoPrefix(raw_token.data(), raw_token.size(), post_func, output_tokens);
    } else {
      TokenizeImplWithPrefix(raw_token.data(), raw_token.size(), post_func, output_tokens);
    }
  } else {
    auto post_func = [](const char* token_buf, int token_len, int value, const List& output) {
      output.push_back(String(token_buf, token_len));
    };
    if (subwords_prefix_.empty()) {
      TokenizeImplNoPrefix(raw_token.data(), raw_token.size(), post_func, output_tokens);
    } else {
      TokenizeImplWithPrefix(raw_token.data(), raw_token.size(), post_func, output_tokens);
    }
  }
}

void WordPieceTokenizer::tokenize(const unicode_view& raw_token, const List& output_tokens) const {
  auto bytes_tokens = UTF8Encode(raw_token);
  if (lookup_id_) {
    auto post_func = [](const char* token_buf, int token_len, int value, const List& output) {
      output.push_back(value);
    };
    if (subwords_prefix_.empty()) {
      TokenizeImplNoPrefix(bytes_tokens.data(), bytes_tokens.size(), post_func, output_tokens);
    } else {
      TokenizeImplWithPrefix(bytes_tokens.data(), bytes_tokens.size(), post_func, output_tokens);
    }
  } else {
    auto post_func = [](const char* token_buf, int token_len, int value, const List& output) {
      output.push_back(UTF8Decode(token_buf, token_len));
    };
    if (subwords_prefix_.empty()) {
      TokenizeImplNoPrefix(bytes_tokens.data(), bytes_tokens.size(), post_func, output_tokens);
    } else {
      TokenizeImplWithPrefix(bytes_tokens.data(), bytes_tokens.size(), post_func, output_tokens);
    }
  }
}

void WordPieceTokenizer::tokenize(const List& sentence, const List& output_tokens) const {
  output_tokens.reserve(output_tokens.size() + sentence.size() + 4);
  for (auto& item : sentence) {
    switch (item.type_code()) {
      case TypeIndex::kRuntimeString: {
        tokenize(item.AsNoCheck<string_view>(), output_tokens);
      } break;
      case TypeIndex::kRuntimeUnicode: {
        tokenize(item.AsNoCheck<unicode_view>(), output_tokens);
      } break;
      default: {
        MXCHECK(false) << "[WordPieceTokenizer] unsupported data type: " << item.type_name();
      }
    }
  }
}

void WordPieceTokenizer::tokenize(const List& sentence,
                                  const List& output_tokens,
                                  const List& output_lens) const {
  output_tokens.reserve(output_tokens.size() + sentence.size() + 4);
  output_lens.reserve(output_lens.size() + sentence.size() + 4);
  for (auto& item : sentence) {
    auto last_size = output_tokens.size();
    switch (item.type_code()) {
      case TypeIndex::kRuntimeString: {
        tokenize(item.AsNoCheck<string_view>(), output_tokens);
      } break;
      case TypeIndex::kRuntimeUnicode: {
        tokenize(item.AsNoCheck<unicode_view>(), output_tokens);
      } break;
      default: {
        MXCHECK(false) << "[WordPieceTokenizer] unsupported data type: List[" << item.type_name()
                       << "]";
      }
    }
    output_lens.push_back(output_tokens.size() - last_size);
  }
}

RTValue WordPieceTokenizer::tokenize(PyArgs args) {
  MXCHECK_EQ(args.size(), 1) << "[WordPieceTokenizer::tokenize] Expect 1 arguments but get "
                             << args.size();
  List output_tokens;
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      this->tokenize(args[0].AsObjectViewNoCheck<List>().data(), output_tokens);
    } break;
    default: {
      MXCHECK(false) << "[WordPieceTokenizer] unsupported data type: " << args[0].type_name();
    } break;
  }
  return output_tokens;
}

RTValue WordPieceTokenizer::tokenize_with_meta(PyArgs args) {
  MXCHECK_EQ(args.size(), 1)
      << "[WordPieceTokenizer::tokenize_with_meta] Expect 1 arguments but get " << args.size();
  List output_tokens;
  List output_lens;
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      this->tokenize(args[0].AsObjectViewNoCheck<List>().data(), output_tokens, output_lens);
    } break;
    default: {
      MXCHECK(false) << "[WordPieceTokenizer] unsupported data type: " << args[0].type_name();
    } break;
  }
  return Tuple::dynamic(output_tokens, output_lens);
}

using text_tokenizer_WordPieceTokenizer = WordPieceTokenizer;
MATX_REGISTER_NATIVE_OBJECT(text_tokenizer_WordPieceTokenizer)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 6) << "[WordPieceTokenizer] Expect 6 arguments but get "
                                 << args.size();
      String vocab_path = commons::details::GetString(args[0], __FILE__, __LINE__);
      bool lookup_id = MATXSCRIPT_TYPE_AS(args[1], int64_t);
      const Any& unk_token = args[2];
      String subwords_prefix = commons::details::GetString(args[3], __FILE__, __LINE__);
      bool skip_empty = MATXSCRIPT_TYPE_AS(args[4], int64_t);
      int64_t max_bytes_per_token = MATXSCRIPT_TYPE_AS(args[5], int64_t);
      return std::make_shared<WordPieceTokenizer>(std::move(vocab_path),
                                                  lookup_id,
                                                  unk_token,
                                                  subwords_prefix,
                                                  skip_empty,
                                                  max_bytes_per_token);
    })
    .RegisterFunction("tokenize",
                      [](void* self, PyArgs args) -> RTValue {
                        return reinterpret_cast<WordPieceTokenizer*>(self)->tokenize(args);
                      })
    .RegisterFunction("tokenize_with_meta", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<WordPieceTokenizer*>(self)->tokenize_with_meta(args);
    });

}  // namespace tokenizer
}  // namespace extension
}  // namespace runtime
}  // namespace matxscript
