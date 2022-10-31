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
#include <matxscript/runtime/container.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(UnicodeView, ConstChars) {
  unicode_view a = U"hello";
  EXPECT_EQ(a, U"hello");
  EXPECT_NE(a, U"hh");
  unicode_view b(U"hello");
  EXPECT_EQ(a, b);
  unicode_view c(U"");
  EXPECT_EQ(c, U"");
  EXPECT_NE(c, U"hh");
  EXPECT_NE(c, a);
  std::cout << a << std::endl;
}

TEST(UnicodeView, std_string) {
  std::basic_string<char32_t> std_s(U"hello");
  unicode_view a = U"hello";
  EXPECT_EQ(a, std_s);
  a = std_s;
  EXPECT_EQ(a, std_s);
  std::basic_string<char32_t> new_std_s((const char32_t*)a.data(), a.size());
  EXPECT_EQ(new_std_s, std_s);
  std::cout << a << std::endl;
}

TEST(UnicodeView, LargeConverter) {
  Unicode raw;
  raw.resize(4096, 'a');
  unicode_view raw_view = raw;
  Unicode copy1 = raw;
  Unicode copy2 = Unicode(raw_view);
  EXPECT_TRUE(copy1.data() == copy2.data());
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_ml.chars ==
              any_copy2.value().data.v_str_store.v_ml.chars);
  unicode_view from_any1 = any_copy1.As<unicode_view>();
  unicode_view from_any2 = any_copy2.As<unicode_view>();
  EXPECT_TRUE(from_any1.data() == from_any2.data());
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_ml.chars ==
              val_copy2.value().data.v_str_store.v_ml.chars);
  unicode_view from_val1 = val_copy1.As<unicode_view>();
  unicode_view from_val2 = val_copy2.As<unicode_view>();
  EXPECT_TRUE(from_val1.data() == from_val2.data());
}

TEST(UnicodeView, MediumConverter) {
  Unicode raw;
  raw.resize(32, 'a');
  unicode_view raw_view = raw;
  Unicode copy1 = raw;
  Unicode copy2 = Unicode(raw_view);
  EXPECT_TRUE(copy1.data() != copy2.data());
  EXPECT_TRUE(copy1 == copy2);
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_ml.chars ==
              any_copy2.value().data.v_str_store.v_ml.chars);
  unicode_view from_any1 = any_copy1.As<unicode_view>();
  unicode_view from_any2 = any_copy2.As<unicode_view>();
  EXPECT_TRUE(from_any1.data() == from_any2.data());
  EXPECT_TRUE(from_any1 == from_any2);
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_ml.chars !=
              val_copy2.value().data.v_str_store.v_ml.chars);
  unicode_view from_val1 = val_copy1.As<unicode_view>();
  unicode_view from_val2 = val_copy2.As<unicode_view>();
  EXPECT_TRUE(from_val1 == from_val2);
  EXPECT_TRUE(from_val1.data() != from_val2.data());
}

TEST(UnicodeView, SmallConverter) {
  Unicode raw;
  raw.resize(1, 'a');
  unicode_view raw_view = raw;
  Unicode copy1 = raw;
  Unicode copy2 = Unicode(raw_view);
  EXPECT_TRUE(copy1.data() != copy2.data());
  EXPECT_TRUE(copy1 == copy2);
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_small_chars !=
              any_copy2.value().data.v_str_store.v_small_chars);
  unicode_view from_any1 = any_copy1.As<unicode_view>();
  unicode_view from_any2 = any_copy2.As<unicode_view>();
  EXPECT_TRUE(from_any1.data() != from_any2.data());
  EXPECT_TRUE(from_any1 == from_any2);
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_small_chars !=
              val_copy2.value().data.v_str_store.v_small_chars);
  unicode_view from_val1 = val_copy1.As<unicode_view>();
  unicode_view from_val2 = val_copy2.As<unicode_view>();
  EXPECT_TRUE(from_val1 == from_val2);
  EXPECT_TRUE(from_val1.data() != from_val2.data());
}

TEST(UnicodeView, Constant) {
  {
    unicode_view raw(U"\u0038\u0038", 2);
    std::cout << raw.category() << std::endl;
  }
  {
    RTView raw(unicode_view(U"\u0038\u0038", 2));
    Unicode s1 = raw.As<Unicode>();
    Unicode s2 = raw.AsNoCheck<Unicode>();
    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;
  }
}

// TODO(mxd) : add more unicode_view test

}  // namespace runtime
}  // namespace matxscript
