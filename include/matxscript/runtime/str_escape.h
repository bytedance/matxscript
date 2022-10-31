// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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
#pragma once

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>

namespace matxscript {
namespace runtime {

/*!
 * \brief Create a String with escape.
 * \param data The data
 * \param size The size of the string.
 * \param use_octal_escape True to use octal escapes instead of hex. If producing C
 *      strings, use octal escapes to avoid ambiguously-long hex escapes.
 * \return the Result string.
 */
runtime::String BytesEscape(const char* data, size_t size, bool use_octal_escape = false);

runtime::String UnicodeEscape(const char32_t* data, size_t size);

inline runtime::String BytesEscape(const runtime::String& val) {
  return BytesEscape(val.data(), val.length(), true);
}

inline runtime::String UnicodeEscape(const runtime::Unicode& val) {
  return UnicodeEscape(val.data(), val.length());
}

}  // namespace runtime
}  // namespace matxscript
