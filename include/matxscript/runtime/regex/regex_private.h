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
#include <cctype>
#include <memory>
#include <sstream>
#include <string>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/regex/regex_pattern.h>

namespace matxscript {
namespace runtime {

class RegexNode : public Object {
 public:
  RegexNode(const string_view& pattern,
            bool ignore_case,
            bool dotall,
            bool extended,
            bool anchored,
            bool ucp);

  List Split(const string_view& input) const;
  List Split(const unicode_view& input) const;
  List Split(const Any& input) const;

  String Replace(const string_view& input, const string_view& repl) const;
  Unicode Replace(const unicode_view& input, const unicode_view& repl) const;
  RTValue Replace(const Any& input, const Any& repl) const;

  Tuple Match(const string_view& input, int64_t offset) const;
  Tuple Match(const unicode_view& input, int64_t offset) const;
  Tuple Match(const Any& input, int64_t offset) const;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeRegex;
  static constexpr const char* _type_key = "Regex";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RegexNode, Object);

 protected:
  std::shared_ptr<regex::RegexPattern> re_;
  String pattern_;

  bool ignore_case_;
  bool dotall_;
  bool extended_;
  bool anchored_;
  bool ucp_;

  unsigned int pcre_opt_;

  friend class Regex;
  friend struct ReprPrinter;
};

}  // namespace runtime
}  // namespace matxscript