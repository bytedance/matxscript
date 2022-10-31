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
#include <matxscript/runtime/generic/generic_funcs_template_args.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(Generic, kernel_object___len__) {
  {
    Unicode a(U"abc");
    ASSERT_EQ(kernel_object___len__(RTView(a)), 3);
    Unicode b(U"\u6d4b\u8bd5\u96c6");
    ASSERT_EQ(kernel_object___len__(RTView(b)), 3);
  }
  {
    String a("abc");
    ASSERT_EQ(kernel_object___len__(RTView(a)), 3);
    String b("\u6d4b\u8bd5\u96c6");
    ASSERT_EQ(kernel_object___len__(RTView(b)), strlen("\u6d4b\u8bd5\u96c6"));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    List b({1, Unicode(U"WORLD"), String("WORLD")});
    ASSERT_EQ(kernel_object___len__(RTView(a)), 3);
    ASSERT_EQ(kernel_object___len__(RTView(b)), 3);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}, {String("HELLO"), String("World")}});
    ASSERT_EQ(kernel_object___len__(RTView(a)), 2);
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    Set b({1, Unicode(U"HELLO"), String("WORLD")});
    Set c({1, Unicode(U"HELLO"), Unicode(U"HELLO")});
    ASSERT_EQ(kernel_object___len__(RTView(a)), 3);
    ASSERT_EQ(kernel_object___len__(RTView(b)), 3);
    ASSERT_EQ(kernel_object___len__(RTView(c)), 2);
  }
}

TEST(Generic, kernel_object___contains__) {
  {
    Unicode con(U"abc");
    ASSERT_TRUE(kernel_object___contains__(con, Unicode(U"bc")));
    ASSERT_TRUE(kernel_object___contains__(con, Unicode(U"ab")));
    ASSERT_FALSE(kernel_object___contains__(con, Unicode(U"ac")));
    con = Unicode(U"\u6d4b\u8bd5\u96c6");
    ASSERT_TRUE(kernel_object___contains__(con, Unicode(U"\u8bd5\u96c6")));
    ASSERT_TRUE(kernel_object___contains__(con, Unicode(U"\u6d4b\u8bd5")));
    ASSERT_FALSE(kernel_object___contains__(con, Unicode(U"\u6d4b\u96c6")));
    ASSERT_FALSE(kernel_object___contains__(con, Unicode(U"ac")));
  }
  {
    String con("abc");
    ASSERT_TRUE(kernel_object___contains__(con, String("bc")));
    ASSERT_TRUE(kernel_object___contains__(con, String("ab")));
    ASSERT_FALSE(kernel_object___contains__(con, String("ac")));
    con = String("\u6d4b\u8bd5\u96c6");
    ASSERT_TRUE(kernel_object___contains__(con, String("\u8bd5\u96c6")));
    ASSERT_TRUE(kernel_object___contains__(con, String("\u6d4b\u8bd5")));
    ASSERT_FALSE(kernel_object___contains__(con, String("\u6d4b\u96c6")));
    ASSERT_FALSE(kernel_object___contains__(con, String("ac")));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    List b({1, Unicode(U"WORLD"), String("WORLD")});
    ASSERT_TRUE(kernel_object___contains__(a, 1));
    ASSERT_TRUE(kernel_object___contains__(a, Unicode(U"HELLO")));
    ASSERT_TRUE(kernel_object___contains__(a, String("HELLO")));
    ASSERT_FALSE(kernel_object___contains__(a, 33));
    ASSERT_FALSE(kernel_object___contains__(a, Unicode(U"HEL")));
    ASSERT_TRUE(kernel_object___contains__(b, 1));
    ASSERT_TRUE(kernel_object___contains__(b, Unicode(U"WORLD")));
    ASSERT_TRUE(kernel_object___contains__(b, String("WORLD")));
    ASSERT_FALSE(kernel_object___contains__(b, 33));
    ASSERT_FALSE(kernel_object___contains__(b, Unicode(U"WORLD1")));
  }
  {
    Dict a({{1, Unicode(U"HELLO")}, {String("HELLO"), String("World")}});
    ASSERT_TRUE(kernel_object___contains__(a, 1));
    ASSERT_TRUE(kernel_object___contains__(a, String("HELLO")));
    ASSERT_FALSE(kernel_object___contains__(a, 33));
    ASSERT_FALSE(kernel_object___contains__(a, Unicode(U"HELLO")));
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_TRUE(kernel_object___contains__(a, 1));
    ASSERT_TRUE(kernel_object___contains__(a, String("HELLO")));
    ASSERT_TRUE(kernel_object___contains__(a, Unicode(U"HELLO")));
    ASSERT_FALSE(kernel_object___contains__(a, 33));
    ASSERT_FALSE(kernel_object___contains__(a, Unicode(U"HELLO1")));
    ASSERT_FALSE(kernel_object___contains__(a, Unicode(U"HELL")));
    ASSERT_FALSE(kernel_object___contains__(a, String("HELL")));
  }
}

TEST(Generic, kernel_object___getitem__) {
  {
    Unicode a(U"abc");
    RTValue r = kernel_object___getitem__(RTView(a), 0);
    Unicode typed_r = r.As<Unicode>();
    ASSERT_EQ(typed_r, Unicode(U"a"));
    ASSERT_THROW(kernel_object___getitem__(RTView(a), 100), Error);
  }
  {
    String a("abc");
    RTValue r = kernel_object___getitem__(RTView(a), 0);
    int64_t typed_r = r.As<int64_t>();
    ASSERT_EQ(typed_r, 'a');
    ASSERT_THROW(kernel_object___getitem__(RTView(a), 100), Error);
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    RTValue r = kernel_object___getitem__(RTView(a), 0);
    int64_t typed_r = r.As<int64_t>();
    ASSERT_EQ(typed_r, 1);
    ASSERT_THROW(kernel_object___getitem__(RTView(a), 100), Error);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}, {String("HELLO"), String("World")}});
    RTValue r = kernel_object___getitem__(RTView(a), String("HELLO"));
    String typed_r = r.As<String>();
    ASSERT_EQ(typed_r, String("World"));
    ASSERT_THROW(kernel_object___getitem__(RTView(a), 100), Error);
    ASSERT_THROW(kernel_object___getitem__(RTView(a), Unicode(U"HELLO")), Error);
  }
}

TEST(Generic, kernel_object___setitem__) {
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    kernel_object___setitem__(a, 0, 33);
    int64_t typed_0 = a.get_item(0).As<int64_t>();
    ASSERT_EQ(typed_0, 33);
    ASSERT_THROW(kernel_object___setitem__(a, 100, 10), Error);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}, {String("HELLO"), String("World")}});
    kernel_object___setitem__(a, 1, String("HELLO"));
    String typed_r = a.get_item(1).As<String>();
    ASSERT_EQ(typed_r, String("HELLO"));
    ASSERT_NO_THROW(kernel_object___setitem__(a, 100, 20));
    ASSERT_NO_THROW(kernel_object___setitem__(a, String("HELLO"), Unicode(U"HELLO")));
  }
}

TEST(Generic, kernel_object___getslice__) {
  {
    Unicode a(U"abc");
    RTValue r = kernel_object___getslice__(a, 0, 1, 1);
    Unicode typed_r = r.As<Unicode>();
    ASSERT_EQ(typed_r, Unicode(U"a"));
    ASSERT_NO_THROW(kernel_object___getslice__(a, 0, 100, 2));
  }
  {
    String a("abc");
    RTValue r = kernel_object___getslice__(a, 0, 1, 1);
    String typed_r = r.As<String>();
    ASSERT_EQ(typed_r, String("a"));
    ASSERT_NO_THROW(kernel_object___getslice__(a, 0, 100, 2));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    RTValue r = kernel_object___getslice__(a, 0, 1, 1);
    List typed_r = r.As<List>();
    ASSERT_EQ(typed_r, List({1}));
    ASSERT_NO_THROW(kernel_object___getslice__(a, 0, 100, 1));
    ASSERT_THROW(kernel_object___getslice__(a, 0, 100, 0), Error);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_THROW(kernel_object___getslice__(a, 0, 1, 1), Error);
  }
  {
    Set a({1, Unicode(U"HELLO")});
    ASSERT_THROW(kernel_object___getslice__(a, 0, 1, 1), Error);
  }
}

TEST(Generic, kernel_object_set_slice) {
  // set_slice is disabled !!!
}

TEST(Generic, kernel_object_append) {
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    kernel_object_append(a, 1);
    int64_t last = a.get_item(-1).As<int64_t>();
    ASSERT_EQ(last, 1);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_THROW(kernel_object_append(a, 1), Error);
  }
  {
    Set a({1, Unicode(U"HELLO")});
    ASSERT_THROW(kernel_object_append(a, 1), Error);
  }
}

TEST(Generic, kernel_object_add) {
  {
    Unicode a(U"abc");
    ASSERT_THROW(kernel_object_add(a, 1), Error);
  }
  {
    String a("abc");
    ASSERT_THROW(kernel_object_add(a, 1), Error);
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_THROW(kernel_object_add(a, 1), Error);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_THROW(kernel_object_add(a, 1), Error);
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    kernel_object_add(a, 2);
    ASSERT_TRUE(a.contains(2));
  }
}

TEST(Generic, kernel_object_extend) {
  {
    Unicode a(U"abc");
    ASSERT_THROW(kernel_object_extend(a, 1), Error);
  }
  {
    String a("abc");
    ASSERT_THROW(kernel_object_extend(a, 1), Error);
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_THROW(kernel_object_extend(a, 1), Error);
    kernel_object_extend(a, List({4, 5}));
    int64_t last = a.get_item(-1).As<int64_t>();
    ASSERT_EQ(last, 5);
    last = a.get_item(-2).As<int64_t>();
    ASSERT_EQ(last, 4);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_THROW(kernel_object_extend(a, 1), Error);
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_THROW(kernel_object_extend(a, 1), Error);
  }
}

TEST(Generic, kernel_object_clear) {
  {
    Unicode a(U"abc");
    ASSERT_THROW(kernel_object_clear(a), Error);
  }
  {
    String a("abc");
    ASSERT_THROW(kernel_object_clear(a), Error);
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_NO_THROW(kernel_object_clear(a));
    ASSERT_TRUE(a.empty());
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_NO_THROW(kernel_object_clear(a));
    ASSERT_TRUE(a.empty());
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_NO_THROW(kernel_object_clear(a));
    ASSERT_TRUE(a.empty());
  }
}

TEST(Generic, kernel_object_find) {
  {
    Unicode a(U"abc");
    ASSERT_NO_THROW(kernel_object_find(a, Unicode(U"a"), 0, 3));
    int64_t pos = kernel_object_find(a, Unicode(U"a"), 0, 3).As<int64_t>();
    ASSERT_EQ(pos, 0);
  }
  {
    String a("abc");
    ASSERT_THROW(kernel_object_find(a, 1, 0, 2), Error);
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_THROW(kernel_object_find(a, 1, 0, 2), Error);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}});
    ASSERT_THROW(kernel_object_find(a, 1, 0, 2), Error);
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_THROW(kernel_object_find(a, 1, 0, 2), Error);
  }
}

TEST(Generic, kernel_object_lower) {
  {
    Unicode raw = U"This is a Test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
    Unicode answer = U"this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
    Unicode res = kernel_object_lower(raw).As<Unicode>();
    std::cout << "raw  : " << raw << std::endl;
    std::cout << "lower: " << res << std::endl;
    ASSERT_EQ(res, answer);
  }
  {
    String raw = "This is a Test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
    String answer = "this is a test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
    String res = kernel_object_lower(raw).As<String>();
    std::cout << "raw  : " << raw << std::endl;
    std::cout << "lower: " << res << std::endl;
    ASSERT_EQ(res, answer);
  }
}

TEST(Generic, kernel_object_upper) {
  {
    Unicode answer = U"THIS IS A TEST, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
    Unicode raw = U"this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
    Unicode res = kernel_object_upper(raw).As<Unicode>();
    std::cout << "raw  : " << raw << std::endl;
    std::cout << "upper: " << res << std::endl;
    ASSERT_EQ(res, answer);
  }
  {
    String answer = "THIS IS A TEST, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
    String raw = "this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
    String res = kernel_object_upper(raw).As<String>();
    std::cout << "raw  : " << raw << std::endl;
    std::cout << "upper: " << res << std::endl;
    ASSERT_EQ(res, answer);
  }
}

TEST(Generic, kernel_object_isdigit) {
  {
    ASSERT_TRUE(kernel_object_isdigit(String("123")).As<bool>());
    ASSERT_FALSE(kernel_object_isdigit(String("123abc")).As<bool>());
  }
  {
    ASSERT_TRUE(kernel_object_isdigit(Unicode(U"123")).As<bool>());
    ASSERT_FALSE(kernel_object_isdigit(Unicode(U"123abc")).As<bool>());
  }
}

TEST(Generic, kernel_object_isalpha) {
  {
    ASSERT_TRUE(kernel_object_isalpha(String("abc")).As<bool>());
    ASSERT_FALSE(kernel_object_isalpha(String("123abc")).As<bool>());
  }
  {
    ASSERT_TRUE(kernel_object_isalpha(Unicode(U"abc")).As<bool>());
    ASSERT_FALSE(kernel_object_isalpha(Unicode(U"123abc")).As<bool>());
  }
}

TEST(Generic, kernel_object_encode) {
  const char* test_case_u8 = R"(Đ Â)";
  const char32_t* test_case_u32 = UR"(Đ Â)";
  const int test_u8_len = std::char_traits<char>::length(test_case_u8);
  const int test_u32_len = std::char_traits<char32_t>::length(test_case_u32);
  String utf8_bytes = kernel_object_encode(Unicode(test_case_u32, test_u32_len)).As<String>();
  ASSERT_EQ(utf8_bytes.size(), test_u8_len);
  ASSERT_EQ(utf8_bytes, std::string(test_case_u8));
}

TEST(Generic, kernel_object_decode) {
  const char* test_case_u8 = R"(Đ Â)";
  const char32_t* test_case_u32 = UR"(Đ Â)";
  const int test_u8_len = std::char_traits<char>::length(test_case_u8);
  const int test_u32_len = std::char_traits<char32_t>::length(test_case_u32);
  Unicode u32_s = kernel_object_decode(String(test_case_u8, test_u8_len)).As<Unicode>();
  ASSERT_EQ(u32_s.size(), test_u32_len);
  ASSERT_EQ(u32_s, Unicode(test_case_u32, test_u32_len));
}

TEST(Generic, kernel_object_split) {
  ASSERT_EQ(kernel_object_split(String("1b2b3"), String("b"), 1),
            (List{String("1"), String("2b3")}));
  ASSERT_EQ(kernel_object_split(Unicode(U"1b2b3"), Unicode(U"b"), 1),
            (List{Unicode(U"1"), Unicode(U"2b3")}));
}

TEST(Generic, kernel_object_join) {
  ASSERT_EQ(kernel_object_join(String(","), List{String("1"), String("2")}), String("1,2"));
  ASSERT_EQ(kernel_object_join(Unicode(U","), List{Unicode(U"1"), Unicode(U"2")}), Unicode(U"1,2"));
}

TEST(Generic, kernel_object_reserve) {
  List l;
  kernel_object_reserve(l, 100);
  ASSERT_EQ(kernel_object_capacity(l), 100);
  Set s;
  kernel_object_reserve(s, 100);
  ASSERT_TRUE(kernel_object_bucket_count(s).As<int64_t>() >= 100);
  Dict d;
  kernel_object_reserve(d, 100);
  ASSERT_TRUE(kernel_object_bucket_count(d).As<int64_t>() >= 100);
}

TEST(Generic, kernel_math_min) {
  List l({1, 2, 3});
  ASSERT_EQ(kernel_math_iterable_min(l), 1);
  Set s({1, 2, 3});
  ASSERT_EQ(kernel_math_iterable_min(s), 1);
  ASSERT_EQ(kernel_math_min(1, 2, 3), 1);
}

TEST(Generic, kernel_math_max) {
  List l({1, 2, 3});
  ASSERT_EQ(kernel_math_iterable_max(l), 3);
  Set s({1, 2, 3});
  ASSERT_EQ(kernel_math_iterable_max(s), 3);
  ASSERT_EQ(kernel_math_max(1, 2, 3), 3);
}

TEST(Generic, kernel_print) {
  List l({1, 2, 3});
  kernel_builtins_print({l});
}

}  // namespace runtime
}  // namespace matxscript
