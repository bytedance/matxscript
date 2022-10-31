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

#include "regex_c_array.h"

#include <cstdint>
#include <string>

#include <matxscript/runtime/container/string.h>

namespace matxscript {
namespace runtime {
namespace regex {

typedef struct {
  const char* source;
  size_t source_len;
  c_array_t* lengths;
  c_array_t* values;
  unsigned int variables;
  unsigned complete_lengths : 1;
  unsigned complete_values : 1;
} regex_sub_script_compile_t;

typedef struct {
  char* ip;
  char* pos;
  char* buf;
  int* captures;
  size_t ncaptures;
  const char* captures_data;
  unsigned skip : 1;
} regex_sub_script_engine_t;

typedef void (*replace_script_code_pt)(regex_sub_script_engine_t* e);

typedef size_t (*replace_script_len_code_pt)(regex_sub_script_engine_t* e);

typedef struct {
  replace_script_code_pt code;
  uintptr_t len;
} regex_sub_script_copy_code_t;

typedef struct {
  replace_script_code_pt code;
  uintptr_t n;
} regex_sub_script_capture_code_t;

class RegexSubHelper {
 public:
  static int Replace(const char* subject,
                     int offset,
                     int count,
                     int* cap,
                     regex_sub_script_compile_t* val,
                     String* buf);

  static int Extract(const char* subject,
                     int offset,
                     int count,
                     int* cap,
                     regex_sub_script_compile_t* val,
                     String* buf);

  static regex_sub_script_compile_t* create_replace_complex_value_t();

  static void destroy_replace_complex_value_t(regex_sub_script_compile_t* v);

  static int Compile(regex_sub_script_compile_t* ccv, String* errmsg);

 private:
  static int replaceScriptCompile(regex_sub_script_compile_t* sc, String* errmsg);

  static int replaceScriptInitArrays(regex_sub_script_compile_t* sc);

  static int replaceScriptAddCopyCode(regex_sub_script_compile_t* sc,
                                      const char* value,
                                      size_t len,
                                      size_t last);

  static int replaceScriptAddCaptureCode(regex_sub_script_compile_t* sc, size_t n);

  static int replaceScriptDone(regex_sub_script_compile_t* sc, String* errmsg);
};

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
