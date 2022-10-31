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
#include <cstring>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include <matxscript/runtime/utf8/encoders.h>

namespace matxscript {
namespace runtime {

namespace {
uint64_t NowNanos() {
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
}

void naive_encode(const char32_t* s_ptr, size_t len, std::basic_string<char>& ret) {
  auto* p_dst = (unsigned char*)(ret.data());
  auto dst_len = utf8_details::NaiveEncoder((uint32_t*)s_ptr, (uint32_t*)s_ptr + len, p_dst);
  ret.resize(dst_len);
}

void greedy_encode(const char32_t* s_ptr, size_t len, std::basic_string<char>& ret) {
  auto* p_dst = (unsigned char*)(ret.data());
  auto dst_len = utf8_details::GreedyEncoder((uint32_t*)s_ptr, (uint32_t*)s_ptr + len, p_dst);
  ret.resize(dst_len);
}

template <typename FuncDecoder>
std::basic_string<char> measure_encode(size_t loop_size,
                                       const std::basic_string<char32_t>& input,
                                       const FuncDecoder& decoder,
                                       const char* prefix) {
  auto begin = NowNanos();
  auto cnt = 0;
  std::basic_string<char> dest(input.size() * 4, 0);
  for (size_t i = 0; i < loop_size; ++i) {
    decoder(input.data(), input.size(), dest);
    cnt += dest.size();
  }
  auto end = NowNanos();

  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);
  std::cout << std::left << std::setfill(' ') << std::setw(25) << prefix << ": run " << loop_size
            << " times, total words: " << cnt << ", using " << std::setprecision(2)
            << double(end - begin) / 1000000.0 << "ms" << std::endl;
  std::cout.copyfmt(old_state);
  return dest;
}

void benchmark(const std::basic_string<char32_t>& s, size_t loop_size, bool check_result) {
  std::cout << "==============================================================================="
            << std::endl;
  std::cout << "input: ";
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);
  for (auto u : s) {
    std::cout << "0x" << std::setfill('0') << std::hex << u << " ";
  }
  std::cout << std::endl;
  std::cout.copyfmt(old_state);

  std::cout << "==============================================================================="
            << std::endl;
  auto r1 = measure_encode(loop_size, s, naive_encode, "baseline");
  auto r2 = measure_encode(loop_size, s, greedy_encode, "greedy_encode");
  std::cout << "==============================================================================="
            << std::endl;
  std::cout << std::endl;
  if (check_result) {
    EXPECT_EQ(r1, r2);
  }
}
}  // namespace

TEST(UTF8Lib, Encoder) {
  std::basic_string<char32_t> small_eng{U"Hello world"};
  std::basic_string<char32_t> small_chn{U"\u4f60\u597d\u4e2d\u56fd"};
  std::basic_string<char32_t> large_eng{U"Add a custom build rule to the generated build system."};
  std::basic_string<char32_t> large_chn{
      U"\u5728\u8fd9\u4e16\u754c\u4e0a\uff0c\u6ca1\u6709\u4e00\u4e2a\u4eba\u662f\u5341\u5168\u5341\u7f8e\u7684\u3002\u6bcf\u4e2a\u4eba\u90fd\u4f1a\u6709\u72af\u9519\u8bef\u7684\u65f6\u5019\uff0c\u4f46\u662f\u6bcf\u4e2a\u4eba\u4e5f\u6709\u8fdb\u6b65\u7684\u65f6\u5019"};
  std::basic_string<char32_t> mixed_data{
      U"\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u006a\u3131\u3134\u3137\u3139\u3141\u3142\u3145\u3147\u3148\u314a\u314b\u314c\u314d\u314e\u007a"};
  size_t loop_size = 100000;

  // test memory overflow
  unsigned char buffer1[16];
  unsigned char buffer2[16];
  for (uint32_t i = 0; i <= 0x10FFFF; ++i) {
    auto* s_ptr = &i;
    auto c1 = utf8_details::NaiveCountBytesSize(s_ptr, s_ptr + 1);
    auto c2 = utf8_details::NaiveEncoder(s_ptr, s_ptr + 1, buffer1);
    EXPECT_EQ(c1, c2);

    auto c3 = utf8_details::GreedyCountBytesSize(s_ptr, s_ptr + 1);
    auto c4 = utf8_details::GreedyEncoder(s_ptr, s_ptr + 1, buffer2);
    EXPECT_EQ(c3, c4);

    EXPECT_EQ(c1, c3);
    EXPECT_EQ(0, std::memcmp(buffer1, buffer2, c1));
  }
  for (uint32_t i = 0x7FFFFFFFu - 1000u; i <= 0x7FFFFFFFu + 1000u; ++i) {
    auto* s_ptr = &i;
    auto c1 = utf8_details::NaiveCountBytesSize(s_ptr, s_ptr + 1);
    auto c2 = utf8_details::NaiveEncoder(s_ptr, s_ptr + 1, buffer1);
    EXPECT_EQ(c1, c2);

    auto c3 = utf8_details::GreedyCountBytesSize(s_ptr, s_ptr + 1);
    auto c4 = utf8_details::GreedyEncoder(s_ptr, s_ptr + 1, buffer2);
    EXPECT_EQ(c3, c4);

    EXPECT_EQ(c1, c3);
    EXPECT_TRUE((0 == std::memcmp(buffer1, buffer2, c1)));
  }

  // bench and check result for normal data
  benchmark(small_eng, loop_size, true);
  benchmark(small_chn, loop_size, true);
  benchmark(large_eng, loop_size, true);
  benchmark(large_chn, loop_size, true);
  benchmark(mixed_data, loop_size, true);
}

}  // namespace runtime
}  // namespace matxscript
