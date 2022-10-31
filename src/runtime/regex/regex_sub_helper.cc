// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file is inspired by openresty.
 * https://github.com/openresty/lua-nginx-module/blob/master/src/ngx_http_lua_regex.c
 *
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
#include <matxscript/runtime/regex/regex_sub_helper.h>

#include <matxscript/runtime/str_printf.h>

namespace matxscript {
namespace runtime {
namespace regex {

static size_t replace_script_copy_capture_len_code(regex_sub_script_engine_t* e) {
  int* cap = nullptr;
  size_t n = 0;
  regex_sub_script_capture_code_t* code = nullptr;

  code = (regex_sub_script_capture_code_t*)e->ip;
  e->ip += sizeof(regex_sub_script_capture_code_t);
  n = code->n;
  if (n < e->ncaptures) {
    cap = e->captures;
    return cap[n + 1] - cap[n];
  }
  return 0;
}

static size_t replace_script_copy_len_code(regex_sub_script_engine_t* e) {
  regex_sub_script_copy_code_t* code = nullptr;
  code = (regex_sub_script_copy_code_t*)e->ip;
  e->ip += sizeof(regex_sub_script_copy_code_t);
  return code->len;
}

static void replace_script_copy_code(regex_sub_script_engine_t* e) {
  regex_sub_script_copy_code_t* code = nullptr;
  code = (regex_sub_script_copy_code_t*)e->ip;
  if (!e->skip) {
    memcpy(e->pos, e->ip + sizeof(regex_sub_script_copy_code_t), code->len);
    e->pos = e->pos + code->len;
  }
  e->ip += sizeof(regex_sub_script_copy_code_t) +
           ((code->len + sizeof(uintptr_t) - 1) & ~(sizeof(uintptr_t) - 1));
}

static void replace_script_copy_capture_code(regex_sub_script_engine_t* e) {
  int* cap = nullptr;
  const char* p = nullptr;
  char* pos = nullptr;
  size_t n = 0;
  regex_sub_script_capture_code_t* code = nullptr;

  code = (regex_sub_script_capture_code_t*)e->ip;
  e->ip += sizeof(regex_sub_script_capture_code_t);
  n = code->n;
  pos = e->pos;
  if (n < e->ncaptures) {
    cap = e->captures;
    p = e->captures_data;
    e->pos = (char*)memcpy(e->pos, &p[cap[n]], cap[n + 1] - cap[n]);
    e->pos = e->pos + cap[n + 1] - cap[n];
  }
}

regex_sub_script_compile_t* RegexSubHelper::create_replace_complex_value_t() {
  regex_sub_script_compile_t* v = nullptr;
  v = (regex_sub_script_compile_t*)malloc(sizeof(regex_sub_script_compile_t));
  if (v == nullptr) {
    throw std::bad_alloc();
  }
  memset(v, 0, sizeof(regex_sub_script_compile_t));
  return v;
}

void RegexSubHelper::destroy_replace_complex_value_t(regex_sub_script_compile_t* v) {
  if (v) {
    if (v->values) {
      c_array_destroy(v->values);
      v->values = nullptr;
    }
    if (v->lengths) {
      c_array_destroy(v->lengths);
      v->lengths = nullptr;
    }
    free(v);
  };
}

int RegexSubHelper::Compile(regex_sub_script_compile_t* ccv, String* errmsg) {
  const char* v = nullptr;
  size_t i = 0;
  size_t n = 0;
  size_t nv = 0;
  c_array_t* lengths = nullptr;
  c_array_t* values = nullptr;
  v = ccv->source;
  nv = 0;
  i = 0;
  while (i < ccv->source_len) {
    if (v[i] == '$') {
      nv++;
    }
    ++i;
  }
  ccv->lengths = nullptr;
  ccv->values = nullptr;
  if (nv == 0) {
    return 1;
  }
  n = nv * (2 * sizeof(regex_sub_script_copy_code_t) + sizeof(regex_sub_script_capture_code_t)) +
      sizeof(uintptr_t);
  lengths = c_array_create(n, 1);
  if (!lengths) {
    StringPrintf(errmsg, "bad alloc");
    return 0;
  }
  n = (nv * (2 * sizeof(regex_sub_script_copy_code_t) + sizeof(regex_sub_script_capture_code_t)) +
       sizeof(uintptr_t) + sizeof(uintptr_t) - 1) &
      ~(sizeof(uintptr_t) - 1);

  values = c_array_create(n, 1);
  if (!values) {
    StringPrintf(errmsg, "bad alloc");
    return 0;
  }
  ccv->lengths = lengths;
  ccv->values = values;
  ccv->complete_lengths = 1;
  ccv->complete_values = 1;
  ccv->variables = nv;
  if (replaceScriptCompile(ccv, errmsg) != 1) {
    c_array_destroy(lengths);
    c_array_destroy(values);
    ccv->lengths = nullptr;
    ccv->values = nullptr;
    return 0;
  }
  return 1;
}

int RegexSubHelper::Replace(const char* subject,
                            int offset,
                            int count,
                            int* cap,
                            regex_sub_script_compile_t* val,
                            String* buf) {
  size_t len = 0;
  char* p = nullptr;
  replace_script_code_pt code = nullptr;
  replace_script_len_code_pt lcode = nullptr;
  regex_sub_script_engine_t e;

  if (val->lengths == nullptr) {
    buf->append(&subject[offset], cap[0] - offset);
    buf->append(val->source, val->source_len);
    return 1;
  }
  memset(&e, 0, sizeof(regex_sub_script_engine_t));
  e.ncaptures = (size_t)count * 2;
  e.captures = cap;
  e.captures_data = subject;
  e.ip = (char*)val->lengths->elts;
  len = 0;
  while (*(uintptr_t*)e.ip) {
    lcode = *(replace_script_len_code_pt*)e.ip;
    len += lcode(&e);
  }
  p = (char*)malloc(len + 1);
  if (p == nullptr) {
    return 0;
  }
  e.ip = (char*)val->values->elts;
  e.pos = p;

  while (*(uintptr_t*)e.ip) {
    code = *(replace_script_code_pt*)e.ip;
    code((regex_sub_script_engine_t*)&e);
  }
  buf->append(&subject[offset], cap[0] - offset);
  buf->append(p, len);
  free(p);
  return 1;
}

int RegexSubHelper::Extract(const char* subject,
                            int offset,
                            int count,
                            int* cap,
                            regex_sub_script_compile_t* val,
                            String* buf) {
  size_t len = 0;
  char* p = nullptr;
  replace_script_code_pt code = nullptr;
  replace_script_len_code_pt lcode = nullptr;
  regex_sub_script_engine_t e;

  if (val->lengths == nullptr) {
    buf->append(val->source, val->source_len);
    return 1;
  }
  memset(&e, 0, sizeof(regex_sub_script_engine_t));
  e.ncaptures = (size_t)count * 2;
  e.captures = cap;
  e.captures_data = subject;
  e.ip = (char*)val->lengths->elts;
  len = 0;
  while (*(uintptr_t*)e.ip) {
    lcode = *(replace_script_len_code_pt*)e.ip;
    len += lcode(&e);
  }
  p = (char*)malloc(len);
  if (p == nullptr) {
    return 0;
  }
  e.ip = (char*)val->values->elts;
  e.pos = p;

  while (*(uintptr_t*)e.ip) {
    code = *(replace_script_code_pt*)e.ip;
    code((regex_sub_script_engine_t*)&e);
  }
  buf->append(p, len);
  free(p);
  return 1;
}

int RegexSubHelper::replaceScriptCompile(regex_sub_script_compile_t* sc, String* errmsg) {
  char ch = '\0';
  const char* name = nullptr;
  size_t name_len = 0;
  size_t i = 0;
  size_t bracket = 0;
  unsigned num_var = 0;
  size_t n = 0;
  const char* errfmt = nullptr;

  if (replaceScriptInitArrays(sc) != 1) {
    return 0;
  }
  for (i = 0; i < sc->source_len; /* void */) {
    name_len = 0;
    if (sc->source[i] == '$') {
      ++i;
      if (sc->source[i] == '\0') {
        errfmt = "invalid capturing variable name found in \"%s\"";
        StringPrintf(errmsg, errfmt, sc->source);
        return 0;
      }
      if (sc->source[i] == '$') {
        name = &sc->source[i];
        ++i;
        ++name_len;
        if (replaceScriptAddCopyCode(sc, name, name_len, (size_t)(i == sc->source_len)) != 1) {
          return 0;
        }
        continue;
      }
      if (sc->source[i] >= '0' && sc->source[i] <= '9') {
        num_var = 1;
        n = 0;
      } else {
        num_var = 0;
      }
      if (sc->source[i] == '{') {
        bracket = 1;
        if (sc->source[++i] == '\0') {
          errfmt = "invalid capturing variable name found in \"%s\"";
          StringPrintf(errmsg, errfmt, sc->source);
          return 0;
        }
        if (sc->source[i] >= '0' && sc->source[i] <= '9') {
          num_var = 1;
          n = 0;
        }
        name = &sc->source[i];
      } else {
        bracket = 0;
        name = &sc->source[i];
      }
      for (/* void */; i < sc->source_len; i++, ++name_len) {
        ch = sc->source[i];
        if (ch == '}' && bracket) {
          i++;
          bracket = 0;
          break;
        }
        if (num_var) {
          if (ch >= '0' && ch <= '9') {
            n = n * 10 + (ch - '0');
            continue;
          }
          break;
        }
        /* not a number variable like $1, $2, etc */
        if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
            ch == '_') {
          continue;
        }
        break;
      }
      if (bracket) {
        errfmt =
            "the closing bracket in \"%c\" "
            "variable is missing";
        StringPrintf(errmsg, errfmt, *name);
        return 0;
      }
      if (name_len == 0) {
        errfmt = "lua script: invalid capturing variable name found in \"%*s\"";
        StringPrintf(errmsg, errfmt, sc->source_len, sc->source);
        return 0;
      }
      if (!num_var) {
        errfmt =
            "attempt to use named capturing variable "
            "\"%c\" (named captures not supported yet)";
        StringPrintf(errmsg, errfmt, *name);
        return 0;
      }
      sc->variables++;
      if (replaceScriptAddCaptureCode(sc, n) != 1) {
        return 0;
      }
      continue;
    }
    name = &sc->source[i];
    while (i < sc->source_len) {
      if (sc->source[i] == '$') {
        break;
      }
      i++;
      name_len++;
    }
    if (replaceScriptAddCopyCode(sc, name, name_len, (size_t)(i == sc->source_len)) != 1) {
      return 0;
    }
  }
  return replaceScriptDone(sc, errmsg);
}

int RegexSubHelper::replaceScriptAddCopyCode(regex_sub_script_compile_t* sc,
                                             const char* value,
                                             size_t len,
                                             size_t last) {
  size_t size = 0;
  regex_sub_script_copy_code_t* code = nullptr;
  code = (regex_sub_script_copy_code_t*)c_array_push_n(sc->lengths,
                                                       sizeof(regex_sub_script_copy_code_t));
  if (code == nullptr) {
    return 0;
  }
  code->code = (replace_script_code_pt)replace_script_copy_len_code;
  code->len = len;
  size = (sizeof(regex_sub_script_copy_code_t) + len + sizeof(uintptr_t) - 1) &
         ~(sizeof(uintptr_t) - 1);
  code = (regex_sub_script_copy_code_t*)c_array_push_n(sc->values, size);
  if (code == nullptr) {
    return 0;
  }
  code->code = replace_script_copy_code;
  code->len = len;
  memcpy(((unsigned char*)code) + sizeof(regex_sub_script_copy_code_t), value, len);
  return 1;
}

int RegexSubHelper::replaceScriptAddCaptureCode(regex_sub_script_compile_t* sc, size_t n) {
  void* p = nullptr;
  regex_sub_script_capture_code_t* code = nullptr;
  p = c_array_push_n(sc->lengths, sizeof(regex_sub_script_capture_code_t));
  if (p == nullptr) {
    return 0;
  }
  code = (regex_sub_script_capture_code_t*)p;
  code->code = (replace_script_code_pt)replace_script_copy_capture_len_code;
  code->n = 2 * n;
  p = c_array_push_n(sc->values, sizeof(regex_sub_script_capture_code_t));
  if (p == nullptr) {
    return 0;
  }
  code = (regex_sub_script_capture_code_t*)p;
  code->code = replace_script_copy_capture_code;
  code->n = 2 * n;
  return 1;
}

int RegexSubHelper::replaceScriptInitArrays(regex_sub_script_compile_t* sc) {
  size_t n = 0;
  if (sc->lengths == nullptr) {
    n = sc->variables *
            (2 * sizeof(regex_sub_script_copy_code_t) + sizeof(regex_sub_script_capture_code_t)) +
        sizeof(uintptr_t);
    sc->lengths = c_array_create(n, 1);
    if (sc->lengths == nullptr) {
      return 0;
    }
  }
  if (sc->values == nullptr) {
    n = (sc->variables *
             (2 * sizeof(regex_sub_script_copy_code_t) + sizeof(regex_sub_script_capture_code_t)) +
         sizeof(uintptr_t) + sizeof(uintptr_t) - 1) &
        ~(sizeof(uintptr_t) - 1);

    sc->values = c_array_create(n, 1);
    if (sc->values == nullptr) {
      return 0;
    }
  }
  sc->variables = 0;
  return 1;
}

int RegexSubHelper::replaceScriptDone(regex_sub_script_compile_t* sc, String* errmsg) {
  uintptr_t* code = nullptr;
  if (sc->complete_lengths) {
    code = (uintptr_t*)c_array_push_n(sc->lengths, sizeof(uintptr_t));
    if (code == nullptr) {
      StringPrintf(errmsg, "bad_alloc");
      return 0;
    }
    *code = (uintptr_t) nullptr;
  }
  if (sc->complete_values) {
    code = (uintptr_t*)c_array_push_n(sc->values, sizeof(uintptr_t));
    if (code == nullptr) {
      StringPrintf(errmsg, "bad_alloc");
      return 0;
    }
    *code = (uintptr_t) nullptr;
  }
  return 1;
}

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
