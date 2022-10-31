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
#include <matxscript/runtime/regex/regex_helper.h>
#include <matxscript/runtime/regex/regex_pattern.h>

namespace matxscript {
namespace runtime {
namespace regex {

std::unique_ptr<RegexPattern> RegexPattern::Load(const String& pattern,
                                                 String* errmsg,
                                                 unsigned int pcre_opt) {
  auto ptr = std::unique_ptr<RegexPattern>(new RegexPattern());
  ptr->pattern_ = pattern;
  ptr->comp_ = RegexHelper::Compile(ptr->pattern_.c_str(), errmsg, pcre_opt);
  if (ptr->comp_ == nullptr) {
    throw std::runtime_error("Failed to compile regex:" + ptr->pattern_);
  }
  return ptr;
}

RegexPattern::RegexPattern() : comp_(nullptr), pattern_() {
}

RegexPattern::~RegexPattern() {
  if (comp_) {
    RegexHelper::Free(comp_);
  }
}

bool RegexPattern::Find(const string_view& subject,
                        int offset,
                        int* from,
                        int* to,
                        String* errmsg,
                        unsigned int pcre_opt) {
  return RegexHelper::Find(
             comp_, subject.data(), subject.size(), offset, from, to, errmsg, pcre_opt) > 0;
}

bool RegexPattern::Split(const string_view& subject,
                         std::vector<String>* result,
                         String* errmsg,
                         unsigned int pcre_opt) {
  return RegexHelper::Split(comp_, subject.data(), subject.size(), result, errmsg, pcre_opt) > 0;
}

bool RegexPattern::Match(const string_view& subject,
                         int offset,
                         std::vector<String>* match_array,
                         std::unordered_map<String, int>* match_named,
                         String* errmsg,
                         unsigned int pcre_opt) {
  return RegexHelper::Match(comp_,
                            subject.data(),
                            subject.size(),
                            offset,
                            match_array,
                            match_named,
                            errmsg,
                            pcre_opt) > 0;
}

bool RegexPattern::Sub(const string_view& subject,
                       const string_view& repl,
                       String* result,
                       String* errmsg,
                       unsigned int pcre_opt) {
  return RegexHelper::Sub(comp_,
                          subject.data(),
                          subject.size(),
                          repl.data(),
                          repl.size(),
                          result,
                          errmsg,
                          pcre_opt) > 0;
}

bool RegexPattern::GSub(const string_view& subject,
                        const string_view& repl,
                        String* result,
                        String* errmsg,
                        unsigned int pcre_opt) {
  return RegexHelper::GSub(comp_,
                           subject.data(),
                           subject.size(),
                           repl.data(),
                           repl.size(),
                           result,
                           errmsg,
                           pcre_opt) > 0;
}

bool RegexPattern::MatchSub(const string_view& subject,
                            const string_view& repl,
                            String* result,
                            String* errmsg,
                            unsigned int pcre_opt) {
  return RegexHelper::MatchSub(comp_,
                               subject.data(),
                               subject.size(),
                               repl.data(),
                               repl.size(),
                               result,
                               errmsg,
                               pcre_opt) > 0;
}

bool RegexPattern::MatchGSub(const string_view& subject,
                             const string_view& repl,
                             String* result,
                             String* errmsg,
                             unsigned int pcre_opt) {
  return RegexHelper::MatchGSub(comp_,
                                subject.data(),
                                subject.size(),
                                repl.data(),
                                repl.size(),
                                result,
                                errmsg,
                                pcre_opt) > 0;
}

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
