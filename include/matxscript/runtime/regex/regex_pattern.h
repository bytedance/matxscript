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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <matxscript/runtime/container/string.h>

namespace matxscript {
namespace runtime {
namespace regex {

class RegexPattern {
 public:
  virtual ~RegexPattern();

  /**
   * Compile a regular expression pattern, returning a pattern object ptr.
   *
   * @param pattern
   * @param errmsg
   * @param pcre_opt
   *        build-in: PCRE_JAVASCRIPT_COMPAT | PCRE_UTF8
   * @return std::unique_ptr<RegexPattern>
   */
  static std::unique_ptr<RegexPattern> Load(const String& pattern,
                                            String* errmsg = nullptr,
                                            unsigned int pcre_opt = 0);

  /**
   * Scan through string looking for a match to the pattern,
   * returning True and set match range: [from, to).
   * returning False if no match was found.
   *
   * @param subject
   * @param offset
   * @param from
   * @param to
   * @param errmsg
   * @param pcre_opt
   * @return
   */
  bool Find(const string_view& subject,
            int offset = 0,
            int* from = nullptr,
            int* to = nullptr,
            String* errmsg = nullptr,
            unsigned int pcre_opt = 0);

  /**
   * Split the source string by the occurrences of the pattern,
   * returning a list containing the resulting substrings.
   *
   * @param subject
   * @param result
   * @param errmsg
   * @param pcre_opt
   * @return
   */
  bool Split(const string_view& subject,
             std::vector<String>* result,
             String* errmsg = nullptr,
             unsigned int pcre_opt = 0);

  /**
   * Scan through string looking for a match to the pattern,
   * returning True and set match captures.
   * returning False if no match was found.
   *
   * @param subject
   * @param offset
   * @param match_array
   * @param match_named
   * @param errmsg
   * @return
   */
  bool Match(const string_view& subject,
             int offset = 0,
             std::vector<String>* match_array = nullptr,
             std::unordered_map<String, int>* match_named = nullptr,
             String* errmsg = nullptr,
             unsigned int pcre_opt = 0);

  /**
   * Return the string obtained by replacing the leftmost
   * non-overlapping occurrences of the pattern in string by the
   * replacement repl.
   *
   * @param subject
   * @param repl
   * @param result
   * @param errmsg
   * @return
   */
  bool Sub(const string_view& subject,
           const string_view& repl,
           String* result,
           String* errmsg = nullptr,
           unsigned int pcre_opt = 0);

  /**
   * Return the string obtained by replacing the
   * non-overlapping occurrences of the pattern in string by the
   * replacement repl.
   *
   * @param subject
   * @param repl
   * @param result
   * @param errmsg
   * @return
   */
  bool GSub(const string_view& subject,
            const string_view& rep,
            String* result,
            String* errmsg = nullptr,
            unsigned int pcre_opt = 0);

  /**
   *
   * @param subject
   * @param repl
   * @param result
   * @param errmsg
   * @return
   */
  bool MatchSub(const string_view& subject,
                const string_view& repl,
                String* result,
                String* errmsg = nullptr,
                unsigned int pcre_opt = 0);

  /**
   *
   * @param subject
   * @param repl
   * @param result
   * @param errmsg
   * @return
   */
  bool MatchGSub(const string_view& subject,
                 const string_view& rep,
                 String* result,
                 String* errmsg = nullptr,
                 unsigned int pcre_opt = 0);

 private:
  RegexPattern();
  RegexPattern(RegexPattern const&) = delete;
  RegexPattern(RegexPattern&&) = delete;
  RegexPattern& operator=(RegexPattern const&) = delete;
  RegexPattern& operator=(RegexPattern&&) = delete;

 private:
  void* comp_;
  String pattern_;
  String errmsg_;

 public:
  friend class std::unique_ptr<RegexPattern>;
};

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
