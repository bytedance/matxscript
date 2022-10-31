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

extern "C" {
#include "pcre.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <matxscript/runtime/container/string.h>

namespace matxscript {
namespace runtime {
namespace regex {

class RegexHelper {
 public:
  typedef struct {
    pcre* code;
    pcre_extra* extra;
    pcre_extra extra_default;
  } regex_t;

  typedef struct {
    unsigned int options;
    regex_t* regex;
    int ncaptures;
    int captures_len;
    int name_count;
    char* name_table;
    int name_entry_size;
    const char* pattern;
  } regex_compile_t;

 public:
  static inline void* Compile(const char* pattern, String* errmsg, unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = create_regex_compile_t();
    if (!re_comp) {
      throw std::bad_alloc();
    }
    re_comp->pattern = pattern;
    re_comp->options |= pcre_opt;
    bool b = compile(re_comp, errmsg);
    if (b) {
      return re_comp;
    } else {
      destroy_regex_compile_t(re_comp);
      return nullptr;
    }
  }

  static inline void Free(void* comp) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    destroy_regex_compile_t(re_comp);
  }

  static inline int Find(void* comp,
                         const char* subject,
                         int subject_len,
                         int offset = 0,
                         int* from = nullptr,
                         int* to = nullptr,
                         String* errmsg = nullptr,
                         unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return Find(re_comp, subject, subject_len, offset, from, to, errmsg, pcre_opt);
  }

  static inline int Split(void* comp,
                          const char* subject,
                          int subject_len,
                          std::vector<String>* result,
                          String* errmsg = nullptr,
                          unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return Split(re_comp, subject, subject_len, result, errmsg, pcre_opt);
  }

  static inline int Match(void* comp,
                          const char* subject,
                          int subject_len,
                          int offset = 0,
                          std::vector<String>* match_array = nullptr,
                          std::unordered_map<String, int>* match_named = nullptr,
                          String* errmsg = nullptr,
                          unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return Match(re_comp, subject, subject_len, offset, match_array, match_named, errmsg, pcre_opt);
  }

  static int Match(regex_compile_t* re_comp,
                   const char* subject,
                   int subject_len,
                   int offset = 0,
                   std::vector<String>* match_array = nullptr,
                   std::unordered_map<String, int>* match_named = nullptr,
                   String* errmsg = nullptr,
                   unsigned int pcre_opt = 0);

  static int Find(regex_compile_t* re_comp,
                  const char* subject,
                  int subject_len,
                  int offset = 0,
                  int* from = nullptr,
                  int* to = nullptr,
                  String* errmsg = nullptr,
                  unsigned int pcre_opt = 0);

  static int Split(regex_compile_t* re_comp,
                   const char* subject,
                   int subject_len,
                   std::vector<String>* result,
                   String* errmsg = nullptr,
                   unsigned int pcre_opt = 0);

  static inline int Sub(void* comp,
                        const char* subject,
                        int subject_len,
                        const char* rep,
                        int rep_len,
                        String* result,
                        String* errmsg = nullptr,
                        unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return SubHelper(
        re_comp, subject, subject_len, rep, rep_len, result, errmsg, 0, false, pcre_opt);
  }

  static inline int GSub(void* comp,
                         const char* subject,
                         int subject_len,
                         const char* rep,
                         int rep_len,
                         String* result,
                         String* errmsg = nullptr,
                         unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return SubHelper(
        re_comp, subject, subject_len, rep, rep_len, result, errmsg, 1, false, pcre_opt);
  }

  static inline int MatchSub(void* comp,
                             const char* subject,
                             int subject_len,
                             const char* rep,
                             int rep_len,
                             String* result,
                             String* errmsg = nullptr,
                             unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return SubHelper(
        re_comp, subject, subject_len, rep, rep_len, result, errmsg, 0, true, pcre_opt);
  }

  static inline int MatchGSub(void* comp,
                              const char* subject,
                              int subject_len,
                              const char* rep,
                              int rep_len,
                              String* result,
                              String* errmsg = nullptr,
                              unsigned int pcre_opt = 0) {
    regex_compile_t* re_comp = static_cast<regex_compile_t*>(comp);
    return SubHelper(
        re_comp, subject, subject_len, rep, rep_len, result, errmsg, 1, true, pcre_opt);
  }

  static int SubHelper(regex_compile_t* re_comp,
                       const char* subject,
                       int subject_len,
                       const char* rep,
                       int rep_len,
                       String* result,
                       String* errmsg = nullptr,
                       unsigned global = 0,
                       bool match_only = false,
                       unsigned int pcre_opt = 0);

 private:
  static regex_t* create_regex_t();

  static void destroy_regex_t(regex_t* re);

  static regex_compile_t* create_regex_compile_t();

  static void destroy_regex_compile_t(regex_compile_t* rc);

  static bool compile(regex_compile_t* rc, String* errmsg);

  static inline int pcreExec(regex_compile_t* rc,
                             const char* subject,
                             int offset,
                             int subject_length,
                             int captures[],
                             int options);
};

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
