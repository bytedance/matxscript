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
#include <matxscript/runtime/unicodelib/unicode_normal_form.h>

namespace matxscript {
namespace runtime {

namespace UnicodeNormalForm {

MATX_DLL int32_t FromStr(string_view form) noexcept {
  if (form == "NFC") {
    return NFC;
  }
  if (form == "NFKC") {
    return NFKC;
  }
  if (form == "NFD") {
    return NFD;
  }
  if (form == "NFKD") {
    return NFKD;
  }
  return Invalid;
}

MATX_DLL string_view ToStr(int32_t form) noexcept {
  switch (form) {
    case NFC: {
      return "NFC";
    } break;
    case NFKC: {
      return "NFKC";
    } break;
    case NFD: {
      return "NFD";
    } break;
    case NFKD: {
      return "NFKD";
    } break;
    default: {
      return "Invalid";
    } break;
  }
}

}  // namespace UnicodeNormalForm

}  // namespace runtime
}  // namespace matxscript
