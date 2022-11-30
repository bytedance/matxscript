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
#include <memory>

#include <matxscript/runtime/algorithm/cedar.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/file_reader.h>
#include <matxscript/runtime/file_util.h>
#include "cppjieba/Jieba.hpp"
#include "matxscript/runtime/container/string_helper.h"
#include "matxscript/runtime/container/unicode_helper.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/type_helper_macros.h"

#include "common_funcs.h"

namespace matxscript {
namespace runtime {
namespace extension {
namespace jieba {

namespace details {
List std_string_list_to_Unicode_List(const std::vector<std::string>& list_of_words_std) {
  List list_of_words;
  list_of_words.reserve(list_of_words_std.size());
  for (const auto& word : list_of_words_std) {
    list_of_words.push_back(std::move(StringHelper::Decode(word)));
  }
  return list_of_words;
}

List std_string_list_to_String_List(const std::vector<std::string>& list_of_words_std) {
  List list_of_words;
  list_of_words.reserve(list_of_words_std.size());
  for (const auto& word : list_of_words_std) {
    list_of_words.push_back(String(word));
  }
  return list_of_words;
}

}  // namespace details

class CPPJieba {
 public:
  CPPJieba(string_view dict_path,
           string_view model_path,
           string_view user_dict_path,
           string_view idfPath,
           string_view stopWordPath) {
    jieba_ptr = std::make_shared<cppjieba::Jieba>(std::string{dict_path},
                                                  std::string{model_path},
                                                  std::string{user_dict_path},
                                                  std::string{idfPath},
                                                  std::string{stopWordPath});
  }
  virtual ~CPPJieba() = default;

 public:
  RTValue lcut(unicode_view sentence, bool cut_all = false, bool HMM = true);
  RTValue lcut(string_view sentence, bool cut_all = false, bool HMM = true);
  RTValue lcut_for_search(unicode_view sentence, bool HMM = true);
  RTValue lcut_for_search(string_view sentence, bool HMM = true);

 private:
  std::shared_ptr<cppjieba::Jieba> jieba_ptr;
};

RTValue CPPJieba::lcut(unicode_view sentence, bool cut_all, bool HMM) {
  MXCHECK(jieba_ptr != nullptr) << "jieba is not initialized.";
  std::string sentence_std = UnicodeHelper::Encode(sentence);
  std::vector<std::string> list_of_words_std;
  if (cut_all) {
    jieba_ptr->CutAll(sentence_std, list_of_words_std);
  } else {
    jieba_ptr->Cut(sentence_std, list_of_words_std, HMM);
  }
  return details::std_string_list_to_Unicode_List(list_of_words_std);
}

RTValue CPPJieba::lcut(string_view sentence, bool cut_all, bool HMM) {
  MXCHECK(jieba_ptr != nullptr) << "jieba is not initialized.";
  std::string sentence_std{sentence};
  std::vector<std::string> list_of_words_std;
  if (cut_all) {
    jieba_ptr->CutAll(sentence_std, list_of_words_std);
  } else {
    jieba_ptr->Cut(sentence_std, list_of_words_std, HMM);
  }
  return details::std_string_list_to_String_List(list_of_words_std);
}

RTValue CPPJieba::lcut_for_search(unicode_view sentence, bool HMM) {
  MXCHECK(jieba_ptr != nullptr) << "jieba is not initialized.";
  std::string sentence_std = UnicodeHelper::Encode(sentence);
  std::vector<std::string> list_of_words_std;
  jieba_ptr->CutForSearch(sentence_std, list_of_words_std, HMM);
  return details::std_string_list_to_Unicode_List(list_of_words_std);
}

RTValue CPPJieba::lcut_for_search(string_view sentence, bool HMM) {
  MXCHECK(jieba_ptr != nullptr) << "jieba is not initialized.";
  std::string sentence_std{sentence};
  std::vector<std::string> list_of_words_std;
  jieba_ptr->CutForSearch(sentence_std, list_of_words_std, HMM);
  return details::std_string_list_to_String_List(list_of_words_std);
}

using text_cutter_CPPJieba = CPPJieba;
MATX_REGISTER_NATIVE_OBJECT(text_cutter_CPPJieba)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 5) << "[CPPJieba] Expect 5 arguments but get " << args.size();
      String dict_path = commons::details::GetString(args[0], __FILE__, __LINE__);
      String model_path = commons::details::GetString(args[1], __FILE__, __LINE__);
      String user_dict_path = commons::details::GetString(args[2], __FILE__, __LINE__);
      String idfPath = commons::details::GetString(args[3], __FILE__, __LINE__);
      String stopWordPath = commons::details::GetString(args[4], __FILE__, __LINE__);
      return std::make_shared<CPPJieba>(
          dict_path, model_path, user_dict_path, idfPath, stopWordPath);
    })
    .RegisterFunction(
        "lcut",
        [](void* self, PyArgs args) -> RTValue {
          switch (args[0].type_code()) {
            case TypeIndex::kRuntimeUnicode: {
              return reinterpret_cast<CPPJieba*>(self)->lcut(
                  args[0].AsNoCheck<unicode_view>(), args[1].As<bool>(), args[2].As<bool>());
            } break;
            case TypeIndex::kRuntimeString: {
              return reinterpret_cast<CPPJieba*>(self)->lcut(
                  args[0].AsNoCheck<string_view>(), args[1].As<bool>(), args[2].As<bool>());
            } break;
            default: {
              MXCHECK(false) << "[Jieba] unsupported data type: " << args[0].type_name();
            } break;
          }
          return List{};
        })
    .RegisterFunction("lcut_for_search", [](void* self, PyArgs args) -> RTValue {
      switch (args[0].type_code()) {
        case TypeIndex::kRuntimeUnicode: {
          return reinterpret_cast<CPPJieba*>(self)->lcut_for_search(
              args[0].AsNoCheck<unicode_view>(), args[1].As<bool>());
        } break;
        case TypeIndex::kRuntimeString: {
          return reinterpret_cast<CPPJieba*>(self)->lcut_for_search(
              args[0].AsNoCheck<string_view>(), args[1].As<bool>());
        } break;
        default: {
          MXCHECK(false) << "[Jieba] unsupported data type: " << args[0].type_name();
        } break;
      }
      return List{};
    });

}  // namespace jieba
}  // namespace extension
}  // namespace runtime
}  // namespace matxscript
