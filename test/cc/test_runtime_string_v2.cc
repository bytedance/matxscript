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
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/ft_container.h>
#include <iostream>
#include <random>

namespace matxscript {
namespace runtime {

TEST(Unicode, CopyAssign) {
  char32_t medium[128];
  for (auto i = 0; i < 128; ++i) {
    medium[i] = i;
  }
  Unicode s1(medium, 32);
  Unicode s2(medium + 7, 34);
  {
    Unicode s3(medium, 32);
    EXPECT_EQ(s3, s1);
  }
  {
    Unicode s3 = s1;
    s3 = s2;
    EXPECT_EQ(s3, s2);
  }
  {
    Unicode s3 = s2;
    s3 = s1;
    EXPECT_EQ(s3, s1);
  }
  {
    Unicode s3(medium + 1, 34);
    s3 = s2;
    EXPECT_EQ(s3, s2);
  }
}

TEST(String, CopyAssign) {
  char medium[128];
  for (auto i = 0; i < 128; ++i) {
    medium[i] = i;
  }
  String s1(medium, 32);
  String s2(medium + 7, 34);
  {
    String s3(medium, 32);
    EXPECT_EQ(s3, s1);
  }
  {
    String s3 = s1;
    s3 = s2;
    EXPECT_EQ(s3, s2);
  }
  {
    String s3 = s2;
    s3 = s1;
    EXPECT_EQ(s3, s1);
  }
  {
    String s3(medium + 1, 34);
    s3 = s2;
    EXPECT_EQ(s3, s2);
  }
}

TEST(Unicode, LogicOp) {
  Unicode small(U"he"), medium(U"hello123"), large(510, U'h');

  // small == small
  ASSERT_EQ(small, Unicode(U"he"));
  // small != small
  ASSERT_NE(small, Unicode(U"h"));
  // small != medium
  ASSERT_NE(small, medium);
  // small != large
  ASSERT_NE(small, large);
  // small char* == small
  ASSERT_EQ(U"he", small);
  // small char* != small
  ASSERT_NE(U"h", small);
  // small char* != medium
  ASSERT_NE(U"h", medium);
  // small char* != large
  ASSERT_NE(U"h", large);
  // small char* == small char*
  ASSERT_EQ(small, U"he");
  // small char* != small char*
  ASSERT_NE(small, U"h");
  // small char* == medium char*
  ASSERT_NE(small, U"hello123");

  // small < small
  ASSERT_LT(Unicode(U"ab"), Unicode(U"ba"));
  // small < medium
  ASSERT_LT(Unicode(U"ab"), Unicode(U"badef"));
  // small < large
  ASSERT_LT(Unicode(U"ab"), Unicode(510, U'e'));
  // small > small
  ASSERT_GT(Unicode(U"ba"), Unicode(U"ab"));
  // small > medium
  ASSERT_GT(Unicode(U"ba"), Unicode(U"abdef"));
  // small > large
  ASSERT_GT(Unicode(U"ba"), Unicode(510, U'a'));

  // medium
  ASSERT_EQ(medium, Unicode(U"hello123"));
  ASSERT_NE(medium, small);
  ASSERT_NE(medium, Unicode(U"hello12345"));
  ASSERT_NE(medium, large);
  ASSERT_EQ(U"hello123", medium);
  ASSERT_NE(U"hello12345", small);
  ASSERT_NE(U"hello12345", medium);
  ASSERT_NE(U"hello12345", large);
  ASSERT_EQ(medium, U"hello123");
  ASSERT_NE(medium, U"h");
  ASSERT_NE(medium, U"hello12345");

  ASSERT_LT(Unicode(U"abcdef"), Unicode(U"ba"));
  ASSERT_LT(Unicode(U"abcdef"), Unicode(U"badef"));
  ASSERT_LT(Unicode(U"abcdef"), Unicode(510, U'e'));
  ASSERT_GT(Unicode(U"bacdef"), Unicode(U"ab"));
  ASSERT_GT(Unicode(U"bacdef"), Unicode(U"abdef"));
  ASSERT_GT(Unicode(U"bacdef"), Unicode(510, U'a'));

  // large
  ASSERT_EQ(large, Unicode(510, U'h'));
  ASSERT_NE(large, small);
  ASSERT_NE(large, medium);
  ASSERT_NE(large, Unicode(510, U'f'));
  ASSERT_NE(large, U"h");
  ASSERT_NE(large, U"hello12345");

  ASSERT_LT(Unicode(510, U'a'), Unicode(U"ba"));
  ASSERT_LT(Unicode(510, U'a'), Unicode(U"badef"));
  ASSERT_LT(Unicode(510, U'a'), Unicode(510, U'e'));
  ASSERT_GT(Unicode(510, U'f'), Unicode(U"ab"));
  ASSERT_GT(Unicode(510, U'f'), Unicode(U"abdef"));
  ASSERT_GT(Unicode(510, U'f'), Unicode(510, U'a'));
}

TEST(Unicode, PlusOp) {
  // plus with char*
  ASSERT_EQ(U"hell" + Unicode(U"o"), Unicode(U"hello"));
  ASSERT_EQ(Unicode(U"hell") + U"o", Unicode(U"hello"));

  // small + small = small
  ASSERT_EQ(Unicode(U"h") + Unicode(U"e"), Unicode(U"he"));
  // small + small = medium
  ASSERT_EQ(Unicode(U"hel") + Unicode(U"lo"), Unicode(U"hello"));
  // small + medium = medium
  ASSERT_EQ(Unicode(U"hel") + Unicode(U"lo1234567"), Unicode(U"hello1234567"));
  // small + medium = large
  ASSERT_EQ(Unicode(U"hh") + Unicode(63, U'h'), Unicode(65, U'h'));

  // medium + small = medium
  ASSERT_EQ(Unicode(U"lo1234567") + Unicode(U"hel"), Unicode(U"lo1234567hel"));
  // medium + medium = medium
  ASSERT_EQ(Unicode(U"lo1234567") + Unicode(U"hel1234567"), Unicode(U"lo1234567hel1234567"));
  // medium + medium = large
  ASSERT_EQ(Unicode(63, U'h') + Unicode(63, U'h'), Unicode(126, U'h'));
  // medium + medium = large
  ASSERT_EQ(Unicode(63, U'h') + Unicode(510, U'h'), Unicode(573, U'h'));

  // large + small = large
  ASSERT_EQ(Unicode(510, U'h') + Unicode(U"h"), Unicode(511, U'h'));
  // large + medium = large
  ASSERT_EQ(Unicode(510, U'h') + Unicode(U"hhhhhhhh"), Unicode(518, U'h'));
  // large + large = large
  ASSERT_EQ(Unicode(510, U'h') + Unicode(510, U'h'), Unicode(1020, U'h'));
}

TEST(Unicode, len) {
  // small
  ASSERT_EQ(Unicode(U"he").size(), 2);
  ASSERT_EQ(Unicode(UR"(ĐÂ)").size(), 2);
  ASSERT_EQ(Unicode(U"\u4e2d\u6587").size(), 2);
  ASSERT_EQ(Unicode(U"\u4e2d\u6587yc").size(), 4);

  // medium
  ASSERT_EQ(Unicode(U"hehehe").size(), 6);
  ASSERT_EQ(Unicode(UR"(ĐÂĐÂĐÂ)").size(), 6);
  ASSERT_EQ(Unicode(U"\u4e2d\u6587\u4e2d\u6587\u4e2d\u6587").size(), 6);
  ASSERT_EQ(Unicode(U"\u4e2d\u6587yc\u4e2d\u6587yc\u4e2d\u6587yc").size(), 12);

  // large
  ASSERT_EQ(Unicode(510, U'h').size(), 510);
  ASSERT_EQ(Unicode(510, U'\u4e2d').size(), 510);
}

TEST(Unicode, Repeat) {
  // small -> small
  ASSERT_EQ(Unicode(U"he").repeat(-1), Unicode());
  ASSERT_EQ(Unicode(U"he").repeat(0), Unicode());
  ASSERT_EQ(Unicode(U"he").repeat(2), Unicode(U"hehe"));
  // small -> medium
  ASSERT_EQ(Unicode(U"he").repeat(10), Unicode(U"hehehehehehehehehehe"));
  // small -> large
  ASSERT_EQ(Unicode(U"h").repeat(510), Unicode(510, U'h'));

  // medium -> medium
  ASSERT_EQ(Unicode(U"hello123").repeat(2), Unicode(U"hello123hello123"));
  // medium -> large
  ASSERT_EQ(Unicode(63, U'h').repeat(10), Unicode(630, U'h'));
}

TEST(Unicode, get_slice) {
  ASSERT_EQ(Unicode(U"hello").get_item(0), Unicode(U"h"));
  ASSERT_EQ(Unicode(U"hello").get_slice(0, 2), Unicode(U"he"));
  ASSERT_EQ(Unicode(U"hello").get_slice(0, 3, 2), Unicode(U"hl"));
  ASSERT_EQ(Unicode(U"hello").get_slice(3, 2), Unicode(U""));
  ASSERT_EQ(Unicode(U"hello").get_slice(-3, 4), Unicode(U"ll"));
}

TEST(Unicode, contains) {
  ASSERT_TRUE(Unicode(U"hello").contains(U"l"));
  ASSERT_TRUE(Unicode(U"hello").contains(U"ell"));
  ASSERT_TRUE(Unicode(U"\u4e2d\u6587ss").contains(U"\u6587s"));
  ASSERT_FALSE(Unicode(U"hello").contains(U"llx"));
  ASSERT_FALSE(Unicode(U"hello").contains(U"xll"));
  ASSERT_FALSE(Unicode(U"\u4e2d\u6587ss").contains(U"\u6587x"));
}

TEST(Unicode, encode) {
  const char* test_case_u8 = R"(Đ Â)";
  const char32_t* test_case_u32 = UR"(Đ Â)";
  const int test_u8_len = std::char_traits<char>::length(test_case_u8);
  const int test_u32_len = std::char_traits<char32_t>::length(test_case_u32);
  String utf8_bytes = Unicode(test_case_u32, test_u32_len).encode();
  ASSERT_EQ(utf8_bytes.size(), test_u8_len);
  ASSERT_EQ(utf8_bytes, std::string(test_case_u8));
}

TEST(Unicode, Lower) {
  Unicode raw = U"This is a Test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
  Unicode answer = U"this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
  Unicode res = raw.lower();
  std::cout << "raw  : " << raw << std::endl;
  std::cout << "lower: " << res << std::endl;
  ASSERT_EQ(res, answer);
}

TEST(Unicode, Upper) {
  Unicode answer = U"THIS IS A TEST, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
  Unicode raw = U"this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
  Unicode res = raw.upper();
  std::cout << "raw  : " << raw << std::endl;
  std::cout << "upper: " << res << std::endl;
  ASSERT_EQ(res, answer);
}

TEST(Unicode, LowerInvalid) {
  constexpr int max_len = 288;
  unsigned char invalid[max_len];
  std::random_device seeder;
  std::mt19937 rng(seeder());
  std::uniform_int_distribution<int> gen(0, 255);  // uniform, unbiased
  for (int i = 0; i < max_len; ++i) {
    invalid[i] = gen(rng);
  }
  const char* invalid_c = reinterpret_cast<const char*>(invalid);
  for (int i = 1; i < max_len; ++i) {
    std::string raw(invalid_c, i);
    std::string lower_s = UTF8DoLower(raw);
    std::string upper_s = UTF8DoUpper(raw);
  }
  std::cout << "ok, all invalid is pass";
}

TEST(Unicode, LowerInvalidV2) {
  constexpr int max_len = 288;
  unsigned char invalid[max_len];
  invalid[0] = 0b11011111;
  const char* invalid_c = reinterpret_cast<const char*>(invalid);
  for (int i = 1; i < 2; ++i) {
    std::string raw(invalid_c, i);
    std::string lower_s = UTF8DoLower(raw);
    std::string upper_s = UTF8DoUpper(raw);
  }
  std::cout << "ok, all invalid v2 is pass";
}

TEST(Unicode, isalpha) {
  ASSERT_TRUE(Unicode(U"abc").isalpha());
  ASSERT_FALSE(Unicode(U"123abc").isalpha());
}

TEST(Unicode, isdigit) {
  ASSERT_TRUE(Unicode(U"123").isdigit());
  ASSERT_FALSE(Unicode(U"123abc").isdigit());
}

TEST(Unicode, split) {
  ASSERT_EQ(Unicode(U"1 2\t3\n4").split(),
            (List{Unicode(U"1"), Unicode(U"2"), Unicode(U"3"), Unicode(U"4")}));
  ASSERT_EQ(Unicode(U"1 2 3\n4").split(U" "),
            (List{Unicode(U"1"), Unicode(U"2"), Unicode(U"3\n4")}));
  ASSERT_EQ(Unicode(U"1ab2ab3ab4").split(U"ab", 1), (List{Unicode(U"1"), Unicode(U"2ab3ab4")}));

  ASSERT_EQ(Unicode(U"1 2 3\n4").split(Unicode(U" ")),
            (List{Unicode(U"1"), Unicode(U"2"), Unicode(U"3\n4")}));
  ASSERT_EQ(Unicode(U"1ab2ab3ab4").split(Unicode(U"ab"), 1),
            (List{Unicode(U"1"), Unicode(U"2ab3ab4")}));
}

TEST(Unicode, join) {
  ASSERT_EQ(Unicode(U",").join(List{Unicode(U"1")}), (Unicode(U"1")));
  ASSERT_EQ(Unicode(U",").join(List{Unicode(U"1"), Unicode(U"2"), Unicode(U"3")}),
            (Unicode(U"1,2,3")));
  ASSERT_EQ(Unicode(U"").join(List{Unicode(U"1"), Unicode(U"2"), Unicode(U"3")}),
            (Unicode(U"123")));
}

TEST(Unicode, replace) {
  Unicode base(U"this is a test string.");
  Unicode str2(U"n example");
  Unicode str3(U"sample phrase");
  Unicode str4(U"useful.");

  Unicode str = base;
  ASSERT_EQ(str.replace(9, 5, str2), Unicode(U"this is an example string."));
  ASSERT_EQ(str.replace(19, 6, str3, 7, 6), Unicode(U"this is an example phrase."));
  ASSERT_EQ(str.replace(8, 10, U"just a"), Unicode(U"this is just a phrase."));
  ASSERT_EQ(str.replace(8, 6, U"a shorty", 7), Unicode(U"this is a short phrase."));
  ASSERT_EQ(str.replace(22, 1, 3, U'!'), Unicode(U"this is a short phrase!!!"));

  // py replace
  ASSERT_EQ(Unicode(U"aabbaabb").replace(Unicode(U"a"), Unicode(U"c")), Unicode(U"ccbbccbb"));
  ASSERT_EQ(Unicode(U"aabbaabbaabb").replace(Unicode(U"ab"), Unicode(U"c"), 2),
            Unicode(U"acbacbaabb"));
}

TEST(Unicode, insert) {
  Unicode str(U"to be question");
  Unicode str2(U"the ");
  Unicode str3(U"or not to be");

  ASSERT_EQ(str.insert(6, str2), Unicode(U"to be the question"));
  ASSERT_EQ(str.insert(6, str3, 3, 4), Unicode(U"to be not the question"));
  ASSERT_EQ(str.insert(10, U"that is cool", 8), Unicode(U"to be not that is the question"));
  ASSERT_EQ(str.insert(10, U"to be "), Unicode(U"to be not to be that is the question"));
  ASSERT_EQ(str.insert(15, 1, U':'), Unicode(U"to be not to be: that is the question"));
  // lack for the insert signatures
  // ASSERT_EQ(str.insert(str.begin() + 5, U','),
  //           Unicode(U"to be(,) not to be: that is the question"));
  // ASSERT_EQ(str.insert(str.end(), 3, U'.'),
  //           Unicode(U"to be, not to be: that is the question(...)"));
}

TEST(Unicode, endswith) {
  ASSERT_TRUE(Unicode(U"01234567890").endswith(Unicode(U"90")));
  ASSERT_TRUE(Unicode(U"01234567890").endswith(Unicode(U"89"), 0, 10));
  ASSERT_FALSE(Unicode(U"01234567890").endswith(Unicode(U"90"), 0, 10));
  ASSERT_FALSE(Unicode(U"01234567890").endswith(Unicode(U"90"), 9, 10));
  ASSERT_TRUE(Unicode(U"01234567890").endswith(Tuple::dynamic(Unicode(U"90"), Unicode(U"89"))));
  ASSERT_TRUE(Unicode(U"01234567890").endswith(RTValue(Unicode(U"89")), 0, 10));
}

TEST(Unicode, startswith) {
  ASSERT_TRUE(Unicode(U"01234567890").startswith(Unicode(U"01")));
  ASSERT_TRUE(Unicode(U"01234567890").startswith(Unicode(U"12"), 1, 3));
  ASSERT_FALSE(Unicode(U"01234567890").startswith(Unicode(U"01"), 1));
  ASSERT_FALSE(Unicode(U"01234567890").startswith(Unicode(U"90"), 9, 10));
  ASSERT_TRUE(Unicode(U"01234567890").startswith(Tuple::dynamic(Unicode(U"23"), Unicode(U"01"))));
  ASSERT_TRUE(Unicode(U"01234567890").startswith(RTValue(Unicode(U"01")), 0, 10));
}

TEST(Unicode, lstrip) {
  ASSERT_EQ(Unicode(U"  \tabc").lstrip(), Unicode(U"abc"));
  ASSERT_EQ(Unicode(U"abcde").lstrip(Unicode(U"ba")), Unicode(U"cde"));
  ASSERT_EQ(Unicode(U"ab").lstrip(Unicode(U"ba")), Unicode(U""));
}

TEST(Unicode, rstrip) {
  ASSERT_EQ(Unicode(U"abc  \t").rstrip(), Unicode(U"abc"));
  ASSERT_EQ(Unicode(U"abcde").rstrip(Unicode(U"de")), Unicode(U"abc"));
  ASSERT_EQ(Unicode(U"ab").rstrip(Unicode(U"ba")), Unicode(U""));
}

TEST(Unicode, strip) {
  ASSERT_EQ(Unicode(U"  abc  \t").strip(), Unicode(U"abc"));
  ASSERT_EQ(Unicode(U"abcde").strip(Unicode(U"dae")), Unicode(U"bc"));
  ASSERT_EQ(Unicode(U"ab").strip(Unicode(U"ba")), Unicode(U""));
}

TEST(Unicode, count) {
  ASSERT_EQ(Unicode(U"001122001122").count(Unicode(U"0")), 4);
  ASSERT_EQ(Unicode(U"001122001122").count(Unicode(U"0"), 1, 4), 1);
  ASSERT_EQ(Unicode(U"001122001122").count(Unicode(U"01")), 2);
  ASSERT_EQ(Unicode(U"001122001122").count(Unicode(U"01"), 3, 12), 1);
}

// bytes
TEST(String, LogicOp) {
  ASSERT_EQ(String("hello"), String("hello"));
  ASSERT_NE(String("hell"), String("hello"));
  ASSERT_LT(String("abc"), String("abd"));
  ASSERT_GT(String("bc"), String("ac"));
}

TEST(String, PlusOp) {
  ASSERT_EQ("hell" + String("o"), String("hello"));
  ASSERT_EQ(String("hell") + String("o"), String("hello"));
}

TEST(String, len) {
  ASSERT_EQ(String("he").size(), 2);
  ASSERT_EQ(String(R"(ĐÂ)").size(), strlen(R"(ĐÂ)"));
  ASSERT_EQ(String("\u4e2d\u6587").size(), strlen("\u4e2d\u6587"));
  ASSERT_EQ(String("\u4e2d\u6587yc").size(), strlen("\u4e2d\u6587yc"));
}

TEST(String, Repeat) {
  ASSERT_EQ(String("he").repeat(3), String("hehehe"));
  ASSERT_EQ(String("he").repeat(0), String());
  ASSERT_EQ(String("he").repeat(-1), String());
}

TEST(String, get_slice) {
  ASSERT_EQ(String("hello").get_item(0), 'h');
  ASSERT_EQ(String("hello").get_slice(0, 2), String("he"));
  ASSERT_EQ(String("hello").get_slice(0, 3, 2), String("hl"));
  ASSERT_EQ(String("hello").get_slice(3, 2), String(""));
  ASSERT_EQ(String("hello").get_slice(-3, 4), String("ll"));
}

TEST(String, contains) {
  ASSERT_TRUE(String("hello").contains("l"));
  ASSERT_TRUE(String("hello").contains("ell"));
  ASSERT_TRUE(String("\u4e2d\u6587ss").contains("\u6587s"));
  ASSERT_FALSE(String("hello").contains("llx"));
  ASSERT_FALSE(String("hello").contains("xll"));
  ASSERT_FALSE(String("\u4e2d\u6587ss").contains("\u6587x"));
}

TEST(String, decode) {
  const char* test_case_u8 = R"(Đ Â)";
  const char32_t* test_case_u32 = UR"(Đ Â)";
  const int test_u8_len = std::char_traits<char>::length(test_case_u8);
  const int test_u32_len = std::char_traits<char32_t>::length(test_case_u32);
  Unicode u32_s = String(test_case_u8, test_u8_len).decode();
  ASSERT_EQ(u32_s.size(), test_u32_len);
  ASSERT_EQ(u32_s, Unicode(test_case_u32, test_u32_len));
}

TEST(String, Lower) {
  String raw = "This is a Test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
  String answer = "this is a test, Đ Â Ă Ê Ô Ơ Ư Ấ Ắ Ế Ố Ớ Ứ Ầ Ằ Ề Ồ Ờ Ừ Ậ Ặ Ệ Ộ Ợ Ự";
  String res = raw.lower();
  std::cout << "raw  : " << raw << std::endl;
  std::cout << "lower: " << res << std::endl;
  ASSERT_EQ(res, answer);
}

TEST(String, Upper) {
  String answer = "THIS IS A TEST, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
  String raw = "this is a test, đ â ă ê ô ơ ư ấ ắ ế ố ớ ứ ầ ằ ề ồ ờ ừ ậ ặ ệ ộ ợ ự";
  String res = raw.upper();
  std::cout << "raw  : " << raw << std::endl;
  std::cout << "upper: " << res << std::endl;
  ASSERT_EQ(res, answer);
}

TEST(String, isalpha) {
  ASSERT_TRUE(String("abc").isalpha());
  ASSERT_FALSE(String("123abc").isalpha());
}

TEST(String, isdigit) {
  ASSERT_TRUE(String("123").isdigit());
  ASSERT_FALSE(String("123abc").isdigit());
}

/* TODO: String to RTValue
TEST(String, split) {
  ASSERT_EQ(String("1 2\t3\n4").split(),
            (List{String("1"), String("2"), String("3"), String("4")}));
  ASSERT_EQ(String("1 2 3\n4").split(" "),
            (List{String("1"), String("2"), String("3\n4")}));
  ASSERT_EQ(String("1ab2ab3ab4").split("ab", 1), (List{String("1"), String("2ab3ab4")}));

  ASSERT_EQ(String("1 2 3\n4").split(String(" ")),
            (List{String("1"), String("2"), String("3\n4")}));
  ASSERT_EQ(String("1ab2ab3ab4").split(String("ab"), 1),
            (List{String("1"), String("2ab3ab4")}));
}

TEST(String, join) {
  ASSERT_EQ(String(",").join(List{String("1")}), (String("1")));
  ASSERT_EQ(String(",").join(List{String("1"), String("2"), String("3")}),
            (String("1,2,3")));
  ASSERT_EQ(String("").join(List{String("1"), String("2"), String("3")}),
            (String("123")));
}
*/
TEST(String, replace) {
  String base("this is a test string.");
  String str2("n example");
  String str3("sample phrase");
  String str4("useful.");

  String str = base;
  ASSERT_EQ(str.replace(9, 5, str2), String("this is an example string."));
  ASSERT_EQ(str.replace(19, 6, str3, 7, 6), String("this is an example phrase."));
  ASSERT_EQ(str.replace(8, 10, "just a"), String("this is just a phrase."));
  ASSERT_EQ(str.replace(8, 6, "a shorty", 7), String("this is a short phrase."));
  ASSERT_EQ(str.replace(22, 1, 3, '!'), String("this is a short phrase!!!"));

  // py replace
  ASSERT_EQ(String("aabbaabb").replace(String("a"), String("c")), String("ccbbccbb"));
  ASSERT_EQ(String("aabbaabbaabb").replace(String("ab"), String("c"), 2), String("acbacbaabb"));
}

TEST(String, insert) {
  String str("to be question");
  String str2("the ");
  String str3("or not to be");

  ASSERT_EQ(str.insert(6, str2), String("to be the question"));
  ASSERT_EQ(str.insert(6, str3, 3, 4), String("to be not the question"));
  ASSERT_EQ(str.insert(10, "that is cool", 8), String("to be not that is the question"));
  ASSERT_EQ(str.insert(10, "to be "), String("to be not to be that is the question"));
  ASSERT_EQ(str.insert(15, 1, ':'), String("to be not to be: that is the question"));
  // lack for the insert signatures
  // ASSERT_EQ(str.insert(str.begin() + 5, U','),
  //           String("to be(,) not to be: that is the question"));
  // ASSERT_EQ(str.insert(str.end(), 3, U'.'),
  //           String("to be, not to be: that is the question(...)"));
}

TEST(String, endswith) {
  ASSERT_TRUE(String("01234567890").endswith(String("90")));
  ASSERT_TRUE(String("01234567890").endswith(String("89"), 0, 10));
  ASSERT_FALSE(String("01234567890").endswith(String("90"), 0, 10));
  ASSERT_FALSE(String("01234567890").endswith(String("90"), 9, 10));
  // TODO: String to RTValue
  // ASSERT_TRUE(String("01234567890").endswith(List{String("90"), String("89")}));
  // TODO: fix RTValue
  // ASSERT_TRUE(String("01234567890").endswith(RTValue(String("89")), 0, 10));
}

TEST(String, startswith) {
  ASSERT_TRUE(String("01234567890").startswith(String("01")));
  ASSERT_TRUE(String("01234567890").startswith(String("12"), 1, 3));
  ASSERT_FALSE(String("01234567890").startswith(String("01"), 1));
  ASSERT_FALSE(String("01234567890").startswith(String("90"), 9, 10));
  // TODO: String to RTValue
  // ASSERT_TRUE(String("01234567890").startswith(List{String("23"), String("01")}));
  // TODO: fix RTValue
  // ASSERT_TRUE(String("01234567890").startswith(RTValue(String("01")), 0, 10));
}

TEST(String, lstrip) {
  ASSERT_EQ(String("  \tabc").lstrip(), String("abc"));
  ASSERT_EQ(String("abcde").lstrip(String("ba")), String("cde"));
  ASSERT_EQ(String("ab").lstrip(String("ba")), String(""));
}

TEST(String, rstrip) {
  ASSERT_EQ(String("abc  \t").rstrip(), String("abc"));
  ASSERT_EQ(String("abcde").rstrip(String("de")), String("abc"));
  ASSERT_EQ(String("ab").rstrip(String("ba")), String(""));
}

TEST(String, strip) {
  ASSERT_EQ(String("  abc  \t").strip(), String("abc"));
  ASSERT_EQ(String("abcde").strip(String("dae")), String("bc"));
  ASSERT_EQ(String("ab").strip(String("ba")), String(""));
}

TEST(String, count) {
  ASSERT_EQ(String("001122001122").count(String("0")), 4);
  ASSERT_EQ(String("001122001122").count(String("0"), 1, 4), 1);
  ASSERT_EQ(String("001122001122").count(String("01")), 2);
  ASSERT_EQ(String("001122001122").count(String("01"), 3, 12), 1);
}

}  // namespace runtime
}  // namespace matxscript
