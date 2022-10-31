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
#include <gtest/gtest.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/unicodelib/py_unicodedata.h>
#include <matxscript/runtime/utf8_util.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(PyUnicodeData, normalize) {
  auto s1 = UTF8Decode("Spicy Jalape\u00f1o");
  auto s2 = UTF8Decode("Spicy Jalapen\u0303o");
  PyUnicodeData unicodedata;
  auto s2_nfc = unicodedata.normalize(UnicodeNormalForm::NFC, s2);
  EXPECT_EQ(s2_nfc, s1);
  auto s1_nfd = unicodedata.normalize(UnicodeNormalForm::NFD, s1);
  EXPECT_EQ(s1_nfd, s2);
}

TEST(PyUnicodeData, decimal) {
  PyUnicodeData unicodedata;
  long d = unicodedata.decimal('9');
  EXPECT_EQ(d, 9);
  EXPECT_THROW(unicodedata.decimal('a'), ValueError);
}

}  // namespace runtime
}  // namespace matxscript
