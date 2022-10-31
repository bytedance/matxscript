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
#include <iostream>

#include <gtest/gtest.h>

#include <matxscript/runtime/utf8/decoders.h>

namespace matxscript {
namespace runtime {

namespace {
uint64_t NowNanos() {
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
}

void naive_decode(const unsigned char* s_ptr, size_t len, std::basic_string<char32_t>& ret) {
  auto* p_dst = const_cast<char32_t*>(ret.data());
  auto dst_len = utf8_details::NaiveDecoder(s_ptr, s_ptr + len, p_dst);
  ret.resize(dst_len);
}

void dfa_decode(const unsigned char* s_ptr, size_t len, std::basic_string<char32_t>& ret) {
  auto* p_dst = const_cast<char32_t*>(ret.data());
  auto dst_len = utf8_details::DFADecoder::Convert(s_ptr, s_ptr + len, p_dst);
  ret.resize(dst_len);
}

void greedy_table_decode(const unsigned char* s_ptr, size_t len, std::basic_string<char32_t>& ret) {
  auto* p_dst = const_cast<char32_t*>(ret.data());
  auto dst_len = utf8_details::GreedyTableDecoder::Convert(s_ptr, s_ptr + len, p_dst);
  ret.resize(dst_len);
}

template <typename FuncDecoder>
std::basic_string<char32_t> measure_decode(size_t loop_size,
                                           const std::string& input,
                                           const FuncDecoder& decoder,
                                           const char* prefix) {
  auto begin = NowNanos();
  auto cnt = 0;
  std::basic_string<char32_t> dest(input.size(), 0);
  for (size_t i = 0; i < loop_size; ++i) {
    decoder((const unsigned char*)input.data(), input.size(), dest);
    cnt += dest.size();
  }
  auto end = NowNanos();

  std::cout << std::left << std::setfill(' ') << std::setw(25) << prefix << ": run " << loop_size
            << " times, total words: " << cnt << ", using " << std::setprecision(2)
            << double(end - begin) / 1000000.0 << "ms" << std::endl;
  return dest;
}

void benchmark(const std::string& s, size_t loop_size, bool check_result) {
  std::cout << "==============================================================================="
            << std::endl;
  std::cout << "input: " << s << std::endl;
  std::cout << "==============================================================================="
            << std::endl;
  auto r1 = measure_decode(loop_size, s, naive_decode, "baseline");
  auto r2 = measure_decode(loop_size, s, dfa_decode, "dfa_decode");
  auto r3 = measure_decode(loop_size, s, greedy_table_decode, "greedy_table_decode");
  std::cout << "==============================================================================="
            << std::endl;
  std::cout << std::endl;
  if (check_result) {
    EXPECT_EQ(r2, r1);
    EXPECT_EQ(r3, r1);
  }
}
}  // namespace

TEST(UTF8Lib, Decoder) {
  std::string small_eng{"Hello world"};
  std::string small_chn{"\u4f60\u597d\u4e2d\u56fd"};
  std::string large_eng{"Add a custom build rule to the generated build system."};
  std::string large_chn{
      "\u5728\u8fd9\u4e16\u754c\u4e0a\uff0c\u6ca1\u6709\u4e00\u4e2a\u4eba\u662f\u5341\u5168\u5341\u7f8e\u7684\u3002\u6bcf\u4e2a\u4eba\u90fd\u4f1a\u6709\u72af\u9519\u8bef\u7684\u65f6\u5019\uff0c\u4f46\u662f\u6bcf\u4e2a\u4eba\u4e5f\u6709\u8fdb\u6b65\u7684\u65f6\u5019"};
  std::string mixed_data{
      "\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u006a\u3131\u3134\u3137\u3139\u3141\u3142\u3145\u3147\u3148\u314a\u314b\u314c\u314d\u314e\u007a"};
  size_t loop_size = 100000;

  // bench and check result for normal data
  benchmark(small_eng, loop_size, true);
  benchmark(small_chn, loop_size, true);
  benchmark(large_eng, loop_size, true);
  benchmark(large_chn, loop_size, true);
  benchmark(mixed_data, loop_size, true);

  // copy from
  // https://www.php.net/manual/en/reference.pcre.pattern.modifiers.php#54805
  // Valid ASCII
  benchmark("a", loop_size, true);
  // Valid 2 Octet Sequence
  benchmark("\xc3\xb1", loop_size, true);
  // Invalid 2 Octet Sequence
  benchmark("\xc3\x28", loop_size, false);
  // Invalid Sequence Identifier
  benchmark("\xa0\xa1", loop_size, false);
  // Valid 3 Octet Sequence
  benchmark("\xe2\x82\xa1", loop_size, false);
  // Invalid 3 Octet Sequence (in 2nd Octet)
  benchmark("\xe2\x28\xa1", loop_size, false);
  // Invalid 3 Octet Sequence (in 3rd Octet)
  benchmark("\xe2\x82\x28", loop_size, false);
  // Valid 4 Octet Sequence
  benchmark("\xf0\x28\x8c\xbc", loop_size, false);
  // Invalid 4 Octet Sequence (in 2nd Octet)
  benchmark("\xf0\x28\x8c\xbc", loop_size, false);
  // Invalid 4 Octet Sequence (in 3rd Octet)
  benchmark("\xf0\x90\x28\xbc", loop_size, false);
  // Invalid 4 Octet Sequence (in 4th Octet)
  benchmark("\xf0\x28\x8c\x28", loop_size, false);
  // Valid 5 Octet Sequence (but not Unicode!)
  benchmark("\xf8\xa1\xa1\xa1\xa1", loop_size, false);
  // Valid 6 Octet Sequence (but not Unicode!)
  benchmark("\xfc\xa1\xa1\xa1\xa1\xa1", loop_size, false);
}

TEST(UTF8Lib, DecoderCheckLength) {
  const char* const valid_examples[] = {
      "a",
      "Hello world",
      "\u4f60\u597d\u4e2d\u56fd",
      "Add a custom build rule to the generated build system.",
      "\u5728\u8fd9\u4e16\u754c\u4e0a\uff0c\u6ca1\u6709\u4e00\u4e2a\u4eba\u662f\u5341\u5168\u5341\u7f8e\u7684\u3002\u6bcf\u4e2a\u4eba\u90fd\u4f1a\u6709\u72af\u9519\u8bef\u7684\u65f6\u5019\uff0c\u4f46\u662f\u6bcf\u4e2a\u4eba\u4e5f\u6709\u8fdb\u6b65\u7684\u65f6\u5019",
      "\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u006a\u3131\u3134\u3137\u3139\u3141\u3142\u3145\u3147\u3148\u314a\u314b\u314c\u314d\u314e\u007a",

      nullptr,
  };
  const char* const invalid_examples[] = {
      "\xc3\x28",
      "\xa0\xa1",
      "\xe2\x82\xa1",
      "\xe2\x28\xa1",
      "\xe2\x82\x28",
      "\xf0\x28\x8c\xbc",
      "\xf0\x28\x8c\xbc",
      "\xf0\x90\x28\xbc",
      "\xf0\x28\x8c\x28",
      "\xf8\xa1\xa1\xa1\xa1",
      "\xfc\xa1\xa1\xa1\xa1\xa1",
      nullptr,
  };

  // test memory overflow
  for (auto i = 0; valid_examples[i]; ++i) {
    auto s_ptr = valid_examples[i];
    auto len = strlen(s_ptr);
    auto s_u_ptr = (unsigned char*)s_ptr;
    auto c1 = utf8_details::GreedyTableDecoder::CountUnitSize(s_u_ptr, s_u_ptr + len);
    std::unique_ptr<char32_t[]> buffer(new char32_t[len]);
    auto c2 = utf8_details::GreedyTableDecoder::Convert(s_u_ptr, s_u_ptr + len, buffer.get());
    EXPECT_EQ(c1, c2);
  }

  for (auto i = 0; invalid_examples[i]; ++i) {
    auto s_ptr = invalid_examples[i];
    auto len = strlen(s_ptr);
    auto s_u_ptr = (unsigned char*)s_ptr;
    auto c1 = utf8_details::GreedyTableDecoder::CountUnitSize(s_u_ptr, s_u_ptr + len);
    std::unique_ptr<char32_t[]> buffer(new char32_t[len]);
    auto c2 = utf8_details::GreedyTableDecoder::Convert(s_u_ptr, s_u_ptr + len, buffer.get());
    EXPECT_EQ(c1, c2);
  }
}

}  // namespace runtime
}  // namespace matxscript
