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
#include <matxscript/runtime/regex/regex_pattern.h>
#include <matxscript/runtime/regex/regex_ref.h>
#include <matxscript/runtime/registry.h>
#include "pcre.h"

namespace matxscript {
namespace runtime {

TEST(RegexPattern, FFI) {
  Unicode pattern = U"name";
  auto pat = Regex(pattern, false, false, false, false, true);
  const auto* RegexMatcherFunc = FunctionRegistry::Get("runtime.RegexMatch");
  auto result = (*RegexMatcherFunc)(std::initializer_list<RTView>{pat, U"mynameisHE", 0});
  EXPECT_EQ(result, Tuple::dynamic(List{U"name"}, Dict()));
}

TEST(RegexPattern, Match) {
  String subject = "+0x234";
  String pattern = R"(^[+-]?(0b|0B|0o|0O|0x|0X)?(\d+)$)";
  std::vector<String> matched_terms;
  auto pat = regex::RegexPattern::Load(pattern, nullptr);
  bool matched = pat->Match(subject, 0, &matched_terms);
  ASSERT_EQ(matched, true);
  ASSERT_EQ(matched_terms.size(), 3);
  ASSERT_EQ(matched_terms[0], "+0x234");
  ASSERT_EQ(matched_terms[1], "0x");
  ASSERT_EQ(matched_terms[2], "234");
}

TEST(RegexPattern, Find) {
  String subject = "abcdef g xxx";
  String pattern = R"([[:space:]])";
  String errmsg;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  auto found = pat->Find(subject);
  ASSERT_GT(found, 0);  // found
  int from = 0;
  int to = 0;
  found = pat->Find(subject, 0, &from, &to, &errmsg);
  ASSERT_GT(found, 0);  // found
  ASSERT_EQ(from, 6);
  ASSERT_EQ(to, 7);
}

TEST(RegexPattern, Split) {
  String subject = "a'z's";
  String pattern = R"((?<=[a-z]['])(?=[a-z]))";
  String errmsg;
  std::vector<String> spl_res;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  auto found = pat->Split(subject, &spl_res);
  ASSERT_GT(found, 0);  // found
  ASSERT_EQ(spl_res.size(), 3);
  ASSERT_EQ(spl_res[0], "a'");
  ASSERT_EQ(spl_res[1], "z'");
  ASSERT_EQ(spl_res[2], "s");
}

TEST(RegexPattern, SplitSpaceTab) {
  String subject = "hello \t world";
  String pattern = R"([ \t]+)";
  String errmsg;
  std::vector<String> spl_res;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  auto found = pat->Split(subject, &spl_res);
  ASSERT_GT(found, 0);  // found
  ASSERT_EQ(spl_res.size(), 2);
  ASSERT_EQ(spl_res[0], "hello");
  ASSERT_EQ(spl_res[1], "world");
}

TEST(RegexPattern, Find_For_Tokenizer) {
  String subject = "a.\"z";
  String pattern = R"((?<=[a-z])\.(?=[\"a-z])|(?<=[a-z\.])\"(?=[a-z]))";
  String errmsg;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  int from = 2;
  int to = 0;
  pat->Find(subject, 0, &from, &to);
  ASSERT_EQ(from, 1);
  ASSERT_EQ(to, 2);
  pat->Find(subject, 2, &from, &to);
  ASSERT_EQ(from, 2);
  ASSERT_EQ(to, 3);
}

TEST(RegexPattern, Find_For_Caseless) {
  String subject = "ABCD XX xx ==";
  String pattern = R"(xx)";
  String errmsg;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg, PCRE_CASELESS);
  auto found = pat->Find(subject);
  ASSERT_GT(found, 0);  // found
  int from = 0;
  int to = 0;
  found = pat->Find(subject, 0, &from, &to, &errmsg);
  ASSERT_GT(found, 0);  // found
  ASSERT_EQ(from, 5);
  ASSERT_EQ(to, 7);
  found = pat->Find(subject, 7, &from, &to, &errmsg);
  ASSERT_GT(found, 0);  // found
  ASSERT_EQ(from, 8);
  ASSERT_EQ(to, 10);
}

TEST(RegexPattern, GSub_For_PunctuationSymbol) {
  String subject = "abcdefg000xxx";
  String pattern =
      R"((_|-|@|:|\?|!|%|\*|\^|#|\[|\]|\{|\}|\"|\&|/|\\|\.|,|\$|\(|\)|;|<|>|。|·|《|》|：|「|」|“|”|、|…|¥|（|）|\d+))";
  String repl = " $1 ";
  String result;
  String errmsg;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  result.clear();
  pat->GSub(subject, repl, &result, &errmsg);
  std::cout << result << std::endl;
  ASSERT_EQ(result, "abcdefg 000 xxx");
}

TEST(RegexPattern, GSub_For_Tokenizer) {
  String subject = "a.\"z";
  String pattern = R"((?<=[a-z])\.(?=[\"a-z])|(?<=[a-z\.])\"(?=[a-z]))";
  String repl = "#";
  String result;
  String errmsg;
  auto pat = regex::RegexPattern::Load(pattern, &errmsg);
  result.clear();
  pat->GSub(subject, repl, &result, &errmsg);
  std::cout << result << std::endl;
  ASSERT_EQ(result, "a##z");
}

TEST(RegexPattern, GSub_For_Space) {
  String result;
  String errmsg;
  const char* subject_c = "hello\t \rgg\n\twor\ufffdld!\0a\u200da";
  auto subject_size = sizeof("hello\t \rgg\n\twor\ufffdld!\0a\u200da");
  String subject(subject_c, subject_size - 1);
  {
    String pattern = "[ \t\n\r\f\\p{Zs}]";
    String repl = " ";
    auto pat = regex::RegexPattern::Load(pattern, &errmsg);
    result.clear();
    pat->GSub(subject, repl, &result, &errmsg);
  }
  {
    subject = result;
    String pattern = "[\\x{0}\\ufffd\\p{Cc}\\p{Cf}\\p{Mn}]";
    String repl;
    auto pat = regex::RegexPattern::Load(pattern, &errmsg, PCRE_UCP);
    result.clear();
    pat->GSub(subject, repl, &result, &errmsg);
  }

  std::cout << result << std::endl;
  ASSERT_EQ(result, "hello   gg  world!aa");
}

}  // namespace runtime
}  // namespace matxscript
