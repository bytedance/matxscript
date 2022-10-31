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

#include <matxscript/runtime/regex/regex_sub_helper.h>
#include <matxscript/runtime/str_printf.h>
#include <matxscript/runtime/unfixed_buffer.h>

namespace matxscript {
namespace runtime {
namespace regex {

RegexHelper::regex_t* RegexHelper::create_regex_t() {
  regex_t* re = nullptr;

  re = (regex_t*)malloc(sizeof(regex_t));
  if (re) {
    re->code = nullptr;
    re->extra = nullptr;
  }
  return re;
}

void RegexHelper::destroy_regex_t(regex_t* re) {
  if (!re) {
    return;
  }
  if (re->code) {
    pcre_free(re->code);
    re->code = nullptr;
  }
  if (re->extra && re->extra != &re->extra_default) {
#ifdef PCRE_CONFIG_JIT
    pcre_free_study(re->extra);
#else
    pcre_free(re->extra);
#endif
    re->extra = nullptr;
  }
  free(re);
}

RegexHelper::regex_compile_t* RegexHelper::create_regex_compile_t() {
  regex_compile_t* rc = nullptr;
  rc = (regex_compile_t*)malloc(sizeof(regex_compile_t));
  if (!rc) {
    throw std::bad_alloc();
  }
  rc->pattern = nullptr;
  rc->options = 0;
  rc->regex = create_regex_t();
  rc->ncaptures = 0;
  // rc->captures = nullptr;
  rc->captures_len = 0;
  rc->name_count = 0;
  rc->name_entry_size = 0;
  rc->name_table = nullptr;
  if (!(rc->regex)) {
    destroy_regex_compile_t(rc);
    throw std::bad_alloc();
  }
  return rc;
}

void RegexHelper::destroy_regex_compile_t(regex_compile_t* rc) {
  if (!rc) {
    return;
  }
  if (rc->regex) {
    destroy_regex_t(rc->regex);
    rc->regex = nullptr;
  }
  // if (rc->name_table) {
  //   free(rc->name_table);
  //   rc->name_table = nullptr;
  // }
  /*
  if (rc->captures) {
    free(rc->captures);
    rc->captures = nullptr;
  } */
  free(rc);
}

bool RegexHelper::compile(regex_compile_t* rc, String* errmsg) {
  int n = 0;
  int erroffset = 0;
  const char* errstr = nullptr;
  const char* p = nullptr;

  /*************************************************************************
   * Now we are going to compile the regular expression pattern, and handle *
   * and errors that are detected.                                          *
   *************************************************************************/
  rc->options |= PCRE_JAVASCRIPT_COMPAT | PCRE_UTF8 | PCRE_NO_UTF8_CHECK;
  rc->regex->code = pcre_compile(rc->pattern, /* the pattern */
                                 rc->options, /* default options */
                                 &errstr,     /* for error message */
                                 &erroffset,  /* for error offset */
                                 nullptr);    /* use default character tables */

  if (rc->regex->code == nullptr) {
    StringPrintf(errmsg, "PCRE compilation failed at offset %d: %s", erroffset, errstr);
    return false;
  }
#ifdef PCRE_CONFIG_JIT
  // Optimize the regex
  rc->regex->extra = pcre_study(rc->regex->code, PCRE_STUDY_JIT_COMPILE, &errstr);
  if (errstr != nullptr) {
    rc->regex->extra_default.flags = PCRE_EXTRA_MATCH_LIMIT | PCRE_EXTRA_MATCH_LIMIT_RECURSION;
    rc->regex->extra = &rc->regex->extra_default;
  } else {
    rc->regex->extra->flags |= PCRE_EXTRA_MATCH_LIMIT | PCRE_EXTRA_MATCH_LIMIT_RECURSION;
  }
  rc->regex->extra->match_limit = 1000000;
  rc->regex->extra->match_limit_recursion = 100000;
#else
  // Optimize the regex
  rc->regex->extra = pcre_study(rc->regex->code, 0, &errstr);
  if (errstr != nullptr) {
    rc->regex->extra_default.flags = PCRE_EXTRA_MATCH_LIMIT | PCRE_EXTRA_MATCH_LIMIT_RECURSION;
    rc->regex->extra = &rc->regex->extra_default;
  } else {
    rc->regex->extra->flags |= PCRE_EXTRA_MATCH_LIMIT | PCRE_EXTRA_MATCH_LIMIT_RECURSION;
  }
  rc->regex->extra->match_limit = 1000000;
  rc->regex->extra->match_limit_recursion = 100000;
#endif
  n = pcre_fullinfo(rc->regex->code, rc->regex->extra, PCRE_INFO_CAPTURECOUNT, &rc->ncaptures);
  if (n < 0) {
    p = "pcre_fullinfo(\"%s\", PCRE_INFO_CAPTURECOUNT) failed: %d";
    StringPrintf(errmsg, p, rc->pattern, n);
    return false;
  }
  rc->captures_len = (rc->ncaptures + 1) * 3;
  /* rc->captures = (int*)malloc(rc->captures_len * sizeof(int));
  if (!rc->captures) {
    StringPrintf(errmsg,"malloc ovectors failed, size:%d",
                                 rc->captures_len * sizeof(int));
    return false;
  } */
  if (rc->ncaptures == 0) {
    return true;
  }
  n = pcre_fullinfo(rc->regex->code, rc->regex->extra, PCRE_INFO_NAMECOUNT, &rc->name_count);
  if (n < 0) {
    p = "pcre_fullinfo(\"%s\", PCRE_INFO_NAMECOUNT) failed: %d";
    StringPrintf(errmsg, p, rc->pattern, n);
    return false;
  }
  if (rc->name_count == 0) {
    return true;
  }
  n = pcre_fullinfo(
      rc->regex->code, rc->regex->extra, PCRE_INFO_NAMEENTRYSIZE, &rc->name_entry_size);
  if (n < 0) {
    p = "pcre_fullinfo(\"%s\", PCRE_INFO_NAMEENTRYSIZE) failed: %d";
    StringPrintf(errmsg, p, rc->pattern, n);
    return false;
  }
  n = pcre_fullinfo(rc->regex->code, rc->regex->extra, PCRE_INFO_NAMETABLE, &rc->name_table);
  if (n < 0) {
    p = "pcre_fullinfo(\"%s\", PCRE_INFO_NAMETABLE) failed: %d";
    StringPrintf(errmsg, p, rc->pattern, n);
    return false;
  }
  return true;
}

int RegexHelper::pcreExec(regex_compile_t* rc,
                          const char* subject,
                          int offset,
                          int subject_length,
                          int captures[],
                          int options) {
  /*************************************************************************
   * If the compilation succeeded, we call PCRE again, in order to do a     *
   * pattern match against the subject string. This does just ONE match. If *
   * further matching is needed, it will be done below.                     *
   *************************************************************************/
  return pcre_exec(rc->regex->code,   /* the compiled pattern */
                   rc->regex->extra,  /* extra data from study the pattern */
                   subject,           /* the subject string */
                   subject_length,    /* the length of the subject */
                   offset,            /* start at offset 0 in the subject */
                   options,           /* default options */
                   captures,          /* output vector for substring information */
                   rc->captures_len); /* number of elements in the output vector */
}

int RegexHelper::Match(regex_compile_t* re_comp,
                       const char* subject,
                       int subject_len,
                       int offset,
                       std::vector<String>* match_array,
                       std::unordered_map<String, int>* match_named,
                       String* errmsg,
                       unsigned int pcre_opt) {
  if (re_comp == nullptr || subject == nullptr || subject_len <= 0) {
    return -1;
  }
  UnfixedBuffer<int, 3072> cap_buf;
  int* captures = cap_buf.Data(re_comp->captures_len);
  if (!captures) {
    StringPrintf(errmsg, "malloc ovectors failed, size:%d", re_comp->captures_len * sizeof(int));
    return -1;
  }
  int rc = pcreExec(re_comp, subject, offset, subject_len, captures, PCRE_NO_UTF8_CHECK | pcre_opt);
  if (rc == PCRE_ERROR_NOMATCH) {
    StringPrintf(errmsg, "no match");
    return 0;
  } else if (rc < 0) {
    if (rc == PCRE_ERROR_BADOPTION) {
      StringPrintf(errmsg, "pcre_exec failed: PCRE_ERROR_BADOPTION");
    } else {
      StringPrintf(errmsg, "pcre_exec failed: %d", rc);
    }
    return -1;
  } else if (rc == 0) {
    StringPrintf(errmsg, "capture size too small");
    return -1;
  }
  if (match_array == nullptr) {
    return 1;
  }
  int i = 0;
  int n = 0;
  for (i = 0, n = 0; i <= re_comp->ncaptures; i++, n += 2) {
    if (i >= rc || captures[n] < 0) {
      match_array->emplace_back();
    } else {
      match_array->emplace_back(subject + captures[n], (size_t)captures[n + 1] - captures[n]);
    }
  }

  if (re_comp->name_count > 0 && match_named) {
    char* name_entry = nullptr;
    char* name = nullptr;
    for (i = 0; i < re_comp->name_count; i++) {
      name_entry = &re_comp->name_table[i * re_comp->name_entry_size];
      n = (name_entry[0] << 8) | name_entry[1];
      name = (char*)&name_entry[2];
      if (n >= match_array->size()) {
        continue;
      }
      if (re_comp->options & PCRE_DUPNAMES) {
        /* unmatched groups are not stored in tables in DUPNAMES mode */
        if (match_array->at(n).empty()) {
          continue;
        }
        match_named->emplace(std::make_pair(String(name), n));
      } else {
        match_named->emplace(std::make_pair(String(name), n));
      }
    }
  }
  return 1;
}

int RegexHelper::Find(regex_compile_t* re_comp,
                      const char* subject,
                      int subject_len,
                      int offset,
                      int* from,
                      int* to,
                      String* errmsg,
                      unsigned int pcre_opt) {
  if (re_comp == nullptr || subject == nullptr || subject_len <= 0) {
    return -1;
  }
  UnfixedBuffer<int, 3072> cap_buf;
  int* captures = cap_buf.Data(re_comp->captures_len);
  if (!captures) {
    StringPrintf(errmsg, "malloc ovectors failed, size:%d", re_comp->captures_len * sizeof(int));
    return -1;
  }
  int rc = pcreExec(re_comp, subject, offset, subject_len, captures, PCRE_NO_UTF8_CHECK | pcre_opt);
  if (rc == PCRE_ERROR_NOMATCH) {
    StringPrintf(errmsg, "no match");
    return 0;
  } else if (rc < 0) {
    StringPrintf(errmsg, "pcre_exec failed: %d", rc);
    return -1;
  } else if (rc == 0) {
    StringPrintf(errmsg, "capture size too small");
    return -1;
  }
  int s_from = captures[0 * 2];
  int s_to = captures[0 * 2 + 1];
  if (!from || !to) {
    if (s_from < 0 || s_to < 0) {
      return 0;
    } else {
      return 1;
    }
  }
  *from = s_from;
  *to = s_to;
  if (s_from < 0 || s_to < 0) {
    return 0;
  } else {
    return 1;
  }
}

int RegexHelper::Split(regex_compile_t* re_comp,
                       const char* subject,
                       int subject_len,
                       std::vector<String>* result,
                       String* errmsg,
                       unsigned int pcre_opt) {
  if (re_comp == nullptr || subject == nullptr || result == nullptr) {
    StringPrintf(errmsg, "Compile input or result is nullptr");
    return -1;
  }
  UnfixedBuffer<int, 3072> cap_buf;
  int* captures = cap_buf.Data(re_comp->captures_len);
  if (!captures) {
    StringPrintf(errmsg, "malloc ovectors failed, size:%d", re_comp->captures_len * sizeof(int));
    return -1;
  }

  int offset = 0;
  int rc = PCRE_ERROR_NOMATCH;
  int count = 0;
  int cp_offset = 0;
  result->clear();
  result->reserve(subject_len);
  for (;;) {
    rc = pcreExec(re_comp, subject, offset, subject_len, captures, PCRE_NO_UTF8_CHECK | pcre_opt);
    if (rc == PCRE_ERROR_NOMATCH) {
      break;
    } else if (rc < 0) {
      StringPrintf(errmsg, "pcre_exec failed: %d", rc);
      return -1;
    } else if (rc == 0) {
      StringPrintf(errmsg, "capture size too small");
      return -1;
    }
    ++count;
    result->emplace_back(subject + cp_offset, captures[0] - cp_offset);
    cp_offset = captures[1];
    offset = cp_offset;
    if (offset == captures[0]) {
      offset++;
      if (offset > subject_len) {
        break;
      }
    }
  }

  if (count == 0) {
    // no match, just the original subject
    result->clear();
    result->emplace_back(subject, subject_len);
  } else {
    if (cp_offset < subject_len) {
      result->emplace_back(&subject[cp_offset], subject_len - cp_offset);
    }
  }
  return 1;
}

int RegexHelper::SubHelper(regex_compile_t* re_comp,
                           const char* subject,
                           int subject_len,
                           const char* rep,
                           int rep_len,
                           String* result,
                           String* errmsg,
                           unsigned global,
                           bool match_only,
                           unsigned int pcre_opt) {
  regex_sub_script_compile_t* ctpl = nullptr;

  if (re_comp == nullptr || subject == nullptr || result == nullptr) {
    StringPrintf(errmsg, "Compile input or result is nullptr");
    return -1;
  }

  ctpl = RegexSubHelper::create_replace_complex_value_t();
  ctpl->source = rep;
  ctpl->source_len = rep_len;
  if (RegexSubHelper::Compile(ctpl, errmsg) != 1) {
    StringPrintf(errmsg, "failed to Compile the replacement template");
    RegexSubHelper::destroy_replace_complex_value_t(ctpl);
    return -1;
  }
  UnfixedBuffer<int, 3072> cap_buf;
  int* captures = cap_buf.Data(re_comp->captures_len);
  if (!captures) {
    StringPrintf(errmsg, "malloc ovectors failed, size:%d", re_comp->captures_len * sizeof(int));
    return -1;
  }

  int offset = 0;
  int rc = PCRE_ERROR_NOMATCH;
  int count = 0;
  int cp_offset = 0;
  for (;;) {
    rc = pcreExec(re_comp, subject, offset, subject_len, captures, PCRE_NO_UTF8_CHECK | pcre_opt);
    if (rc == PCRE_ERROR_NOMATCH) {
      break;
    } else if (rc < 0) {
      RegexSubHelper::destroy_replace_complex_value_t(ctpl);
      StringPrintf(errmsg, "pcre_exec failed: %d", rc);
      return -1;
    } else if (rc == 0) {
      RegexSubHelper::destroy_replace_complex_value_t(ctpl);
      StringPrintf(errmsg, "capture size too small");
      return -1;
    }
    ++count;
    if (match_only) {
      rc = RegexSubHelper::Extract(subject, cp_offset, rc, captures, ctpl, result);
    } else {
      rc = RegexSubHelper::Replace(subject, cp_offset, rc, captures, ctpl, result);
    }

    if (rc != 1) {
      RegexSubHelper::destroy_replace_complex_value_t(ctpl);
      StringPrintf(errmsg,
                   "failed to eval the template for "
                   "replacement: \"%*s\"",
                   rep_len,
                   rep);
      return -1;
    }
    cp_offset = captures[1];
    offset = cp_offset;
    if (offset == captures[0]) {
      offset++;
      if (offset > subject_len) {
        break;
      }
    }
    if (global) {
      continue;
    }
    break;
  }

  if (count == 0) {
    // no match, just the original subject
    result->clear();
    if (!match_only) {
      result->append(subject, subject_len);
    }
  } else {
    if (cp_offset < subject_len && !match_only) {
      result->append(&subject[cp_offset], subject_len - cp_offset);
    }
  }
  RegexSubHelper::destroy_replace_complex_value_t(ctpl);
  return 1;
}

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
