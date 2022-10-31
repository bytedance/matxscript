// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Author: andrei.alexandrescu@fb.com

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <list>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace runtime {

static const int seed = 201208;
using RandomT = std::mt19937;
static RandomT rng(seed);
static const size_t maxTString = 100;
static const bool avoidAliasing = true;

template <class Integral1, class Integral2>
Integral2 random(Integral1 low, Integral2 up) {
  std::uniform_int_distribution<> range(low, up);
  return range(rng);
}

template <class TString>
void randomTString(TString* toFill, unsigned int maxSize = 1000) {
  assert(toFill);
  toFill->resize(random(0, maxSize));
  for (size_t i = 0; i < toFill->length(); ++i) {
    toFill->at(i) = static_cast<typename TString::value_type>(random('a', 'z'));
  }
}

template <class TString, class Integral>
void Num2TString(TString& str, Integral n) {
  std::string tmp = std::to_string(n);
  str = TString(tmp.begin(), tmp.end());
}

std::list<char> RandomList(unsigned int maxSize) {
  std::list<char> lst(random(0u, maxSize));
  std::list<char>::iterator i = lst.begin();
  for (; i != lst.end(); ++i) {
    *i = random('a', 'z');
  }
  return lst;
}

////////////////////////////////////////////////////////////////////////////////
// Tests begin here
////////////////////////////////////////////////////////////////////////////////

template <class TString>
void clause11_21_4_2_a(TString& test) {
  test.TString::~TString();
  new (&test) TString();
}
template <class TString>
void clause11_21_4_2_b(TString& test) {
  TString test2(test);
  assert(test2 == test);
}
template <class TString>
void clause11_21_4_2_c(TString& test) {
  // Test move constructor. There is a more specialized test, see
  // testMoveCtor test
  TString donor(test);
  TString test2(std::move(donor));
  EXPECT_EQ(test2, test);
  // Technically not required, but all implementations that actually
  // support move will move large strings. Make a guess for 128 as the
  // maximum small string optimization that's reasonable.
  EXPECT_LE(donor.size(), 128);
}
template <class TString>
void clause11_21_4_2_d(TString& test) {
  // Copy constructor with position and length
  const size_t pos = random(0, test.size());
  TString s(test,
            pos,
            random(0, 9) ? random(0, (size_t)(test.size() - pos))
                         : TString::npos);  // test for npos, too, in 10% of the cases
  test = s;
}
template <class TString>
void clause11_21_4_2_e(TString& test) {
  // Constructor from char*, size_t
  const size_t pos = random(0, test.size()), n = random(0, test.size() - pos);
  TString before(test.data(), test.size());
  TString s(test.c_str() + pos, n);
  TString after(test.data(), test.size());
  EXPECT_EQ(before, after);
  test = s;
}
template <class TString>
void clause11_21_4_2_f(TString& test) {
  // Constructor from char*
  const size_t pos = random(0, test.size());
  TString before(test.data(), test.size());
  TString s(test.c_str() + pos);
  TString after(test.data(), test.size());
  EXPECT_EQ(before, after);
  test = s;
}
template <class TString>
void clause11_21_4_2_g(TString& test) {
  // Constructor from size_t, char
  const size_t n = random(0, test.size());
  const auto c = test.front();
  test = TString(n, c);
}
template <class TString>
void clause11_21_4_2_h(TString& test) {
  // Constructors from various iterator pairs
  // Constructor from char*, char*
  TString s1(test.begin(), test.end());
  EXPECT_EQ(test, s1);
  TString s2(test.data(), test.data() + test.size());
  EXPECT_EQ(test, s2);
  // Constructor from other iterators
  std::list<char> lst;
  for (auto c : test) {
    lst.push_back(c);
  }
  TString s3(lst.begin(), lst.end());
  EXPECT_EQ(test, s3);
  // Constructor from char32_t iterators
  std::list<char32_t> lst1;
  for (auto c : test) {
    lst1.push_back(c);
  }
  TString s4(lst1.begin(), lst1.end());
  EXPECT_EQ(test, s4);
  // Constructor from char32_t pointers
  char32_t t[20];
  t[0] = U'a';
  t[1] = U'b';
  TString s5(t, t + 2);
}
template <class TString>
void clause11_21_4_2_i(TString& test) {
  // From initializer_list<char>
  std::initializer_list<typename TString::value_type> il = {'h', 'e', 'l', 'l', 'o'};
  TString s(il);
  test = s;
}
template <class TString>
void clause11_21_4_2_j(TString& test) {
  // Assignment from const TString&
  auto size = random(0, 2000);
  TString s(size, '\0');
  EXPECT_EQ(s.size(), size);
  for (size_t i = 0l; i < s.size(); ++i) {
    s[i] = random('a', 'z');
  }
  test = s;
}
template <class TString>
void clause11_21_4_2_k(TString& test) {
  // Assignment from TString&&
  auto size = random(0, 2000);
  TString s(size, '\0');
  EXPECT_EQ(s.size(), size);
  for (size_t i = 0l; i < s.size(); ++i) {
    s[i] = random('a', 'z');
  }
  test = std::move(s);
  EXPECT_LE(s.size(), 128);
}
template <class TString>
void clause11_21_4_2_l(TString& test) {
  // Assignment from char*
  TString s(random(0, 1000), '\0');
  size_t i = 0;
  for (; i != s.size(); ++i) {
    s[i] = random('a', 'z');
  }
  test = s.c_str();
}
template <class TString>
void clause11_21_4_2_lprime(TString& test) {
  // Aliased assign
  const size_t pos = random(0, test.size());
  if (avoidAliasing) {
    test = TString(test.c_str() + pos);
  } else {
    test = test.c_str() + pos;
  }
}
template <class TString>
void clause11_21_4_2_m(TString& test) {
  // Assignment from char
  using value_type = typename TString::value_type;
  test = random(static_cast<value_type>('a'), static_cast<value_type>('z'));
}
template <class TString>
void clause11_21_4_2_n(TString& test) {
  // Assignment from initializer_list<char>
  std::initializer_list<typename TString::value_type> il = {'h', 'e', 'l', 'l', 'o'};
  test = il;
}

template <class TString>
void clause11_21_4_3(TString& test) {
  // Iterators. The code below should leave test unchanged
  EXPECT_EQ(test.size(), test.end() - test.begin());
  EXPECT_EQ(test.size(), test.rend() - test.rbegin());
  //   EXPECT_EQ(test.size(), test.cend() - test.cbegin());
  //   EXPECT_EQ(test.size(), test.crend() - test.crbegin());

  auto s = test.size();
  test.resize(test.end() - test.begin());
  EXPECT_EQ(s, test.size());
  test.resize(test.rend() - test.rbegin());
  EXPECT_EQ(s, test.size());
}

template <class TString>
void clause11_21_4_4(TString& test) {
  // exercise capacity, size, max_size
  EXPECT_EQ(test.size(), test.length());
  //   EXPECT_LE(test.size(), test.max_size());
  //   EXPECT_LE(test.capacity(), test.max_size());
  EXPECT_LE(test.size(), test.capacity());

  // exercise shrink_to_fit. Nonbinding request so we can't really do
  // much beyond calling it.
  auto copy = test;
  copy.reserve(copy.capacity() * 3);
  copy.shrink_to_fit();
  EXPECT_EQ(copy, test);

  // exercise empty
  std::string empty("empty");
  std::string notempty("not empty");
  if (test.empty()) {
    test = TString(empty.begin(), empty.end());
  } else {
    test = TString(notempty.begin(), notempty.end());
  }
}

template <class TString>
void clause11_21_4_5(TString& test) {
  // exercise element access
  if (!test.empty()) {
    EXPECT_EQ(test[0], test.front());
    EXPECT_EQ(test[test.size() - 1], test.back());
    auto const i = random(0, test.size() - 1);
    EXPECT_EQ(test[i], test.at(i));
    test = test[i];
  }

  EXPECT_THROW(test.at(test.size()), std::out_of_range);
  const TString& const_test(test);
  EXPECT_THROW(const_test.at(test.size()), std::out_of_range);
}

template <class TString>
void clause11_21_4_6_1(TString& test) {
  // 21.3.5 modifiers (+=)
  TString test1;
  randomTString(&test1);
  assert(test1.size() == std::char_traits<typename TString::value_type>::length(test1.c_str()));
  auto len = test.size();
  test += test1;
  EXPECT_EQ(test.size(), test1.size() + len);
  for (size_t i = 0; i < test1.size(); ++i) {
    EXPECT_EQ(test[len + i], test1[i]);
  }
  // aliasing modifiers
  TString test2 = test;
  auto dt = test2.data();
  auto sz = test.c_str();
  len = test.size();
  EXPECT_EQ(memcmp(sz, dt, len), 0);
  TString copy(test.data(), test.size());
  EXPECT_EQ(std::char_traits<typename TString::value_type>::length(test.c_str()), len);
  test += test;
  // test.append(test);
  EXPECT_EQ(test.size(), 2 * len);
  EXPECT_EQ(std::char_traits<typename TString::value_type>::length(test.c_str()), 2 * len);
  for (size_t i = 0; i < len; ++i) {
    EXPECT_EQ(test[i], copy[i]);
    EXPECT_EQ(test[i], test[len + i]);
  }
  len = test.size();
  EXPECT_EQ(std::char_traits<typename TString::value_type>::length(test.c_str()), len);
  // more aliasing
  auto const pos = random(0, test.size());
  EXPECT_EQ(std::char_traits<typename TString::value_type>::length(test.c_str() + pos), len - pos);
  if (avoidAliasing) {
    TString addMe(test.c_str() + pos);
    EXPECT_EQ(addMe.size(), len - pos);
    test += addMe;
  } else {
    test += test.c_str() + pos;
  }
  EXPECT_EQ(test.size(), 2 * len - pos);
  // single char
  len = test.size();
  test += random('a', 'z');
  EXPECT_EQ(test.size(), len + 1);
  // initializer_list
  std::initializer_list<typename TString::value_type> il{'a', 'b', 'c'};
  test += il;
}

template <class TString>
void clause11_21_4_6_2(TString& test) {
  // 21.3.5 modifiers (append, push_back)
  TString s;

  // Test with a small string first
  char c = random('a', 'z');
  s.push_back(c);
  EXPECT_EQ(s[s.size() - 1], c);
  EXPECT_EQ(s.size(), 1);
  s.resize(s.size() - 1);

  randomTString(&s, maxTString);
  test.append(s);
  randomTString(&s, maxTString);
  test.append(s, random(0, s.size()), random(0, maxTString));
  randomTString(&s, maxTString);
  // no append(char*, int)
  // test.append(s.c_str(), random(0, s.size()));
  // randomTString(&s, maxTString);
  // test.append(s.c_str());
  // test.append(random(0, maxTString), random('a', 'z'));
  // std::list<char> lst(RandomList(maxTString));
  // test.append(lst.begin(), lst.end());
  // c = random('a', 'z');
  // test.push_back(c);
  // EXPECT_EQ(test[test.size() - 1], c);
  // // initializer_list
  // std::initializer_list<typename TString::value_type> il{'a', 'b', 'c'};
  // test.append(il);
}

template <class TString>
void clause11_21_4_6_3_a(TString& test) {
  // assign
  TString s;
  randomTString(&s);
  test.assign(s);
  EXPECT_EQ(test, s);
  // move assign
  test.assign(std::move(s));
  EXPECT_LE(s.size(), 128);
}

template <class TString>
void clause11_21_4_6_3_b(TString& test) {
  // assign
  TString s;
  randomTString(&s, maxTString);
  test.assign(s, random(0, s.size()), random(0, maxTString));
}

template <class TString>
void clause11_21_4_6_3_c(TString& test) {
  // assign
  TString s;
  randomTString(&s, maxTString);
  test.assign(s.c_str(), random(0, s.size()));
}

template <class TString>
void clause11_21_4_6_3_d(TString& test) {
  // assign
  // no assign(char*)
  //   TString s;
  //   randomTString(&s, maxTString);
  //   test.assign(s.c_str());
}

template <class TString>
void clause11_21_4_6_3_e(TString& test) {
  // assign
  TString s;
  randomTString(&s, maxTString);
  test.assign(random(0, maxTString), random('a', 'z'));
}

template <class TString>
void clause11_21_4_6_3_f(TString& test) {
  // assign from bidirectional iterator
  std::list<char> lst(RandomList(maxTString));
  test.assign(lst.begin(), lst.end());
}

template <class TString>
void clause11_21_4_6_3_g(TString& test) {
  // assign from aliased source
  test.assign(test);
}

template <class TString>
void clause11_21_4_6_3_h(TString& test) {
  // assign from aliased source
  test.assign(test, random(0, test.size()), random(0, maxTString));
}

template <class TString>
void clause11_21_4_6_3_i(TString& test) {
  // assign from aliased source
  test.assign(test.c_str(), random(0, test.size()));
}

template <class TString>
void clause11_21_4_6_3_j(TString& test) {
  // assign from aliased source
  // no assign(char*)
  //   test.assign(test.c_str());
}

template <class TString>
void clause11_21_4_6_3_k(TString& test) {
  // assign from initializer_list
  std::initializer_list<typename TString::value_type> il{'a', 'b', 'c'};
  test.assign(il);
}

template <class TString>
void clause11_21_4_6_4(TString& test) {
  // insert
  TString s;
  randomTString(&s, maxTString);
  test.insert(random(0, test.size()), s);
  randomTString(&s, maxTString);
  test.insert(random(0, test.size()), s, random(0, s.size()), random(0, maxTString));
  randomTString(&s, maxTString);
  test.insert(random(0, test.size()), s.c_str(), random(0, s.size()));
  randomTString(&s, maxTString);
  test.insert(random(0, test.size()), s.c_str());
  test.insert(random(0, test.size()), random(0, maxTString), random('a', 'z'));
  typename TString::size_type pos = random(0, test.size());
  // no insert(iter, char)
  //   typename TString::iterator res =
  //       test.insert(test.begin() + pos, random('a', 'z'));
  //   EXPECT_EQ(res - test.begin(), pos);
  std::list<char> lst(RandomList(maxTString));
  pos = random(0, test.size());
  // Uncomment below to see a bug in gcc
  //   /*res = */ test.insert(test.begin() + pos, lst.begin(), lst.end());
  // insert from initializer_list
  std::initializer_list<typename TString::value_type> il{'a', 'b', 'c'};
  pos = random(0, test.size());
  // Uncomment below to see a bug in gcc
  //   /*res = */ test.insert(test.begin() + pos, il);

  // Test with actual input iterators
  //   std::stringstream ss;
  //   ss << "hello cruel world";
  //   auto i = std::istream_iterator<char>(ss);
  //   test.insert(test.begin(), i, std::istream_iterator<char>());
}

template <class TString>
void clause11_21_4_6_5(TString& test) {
  // erase and pop_back
  if (!test.empty()) {
    test.erase(random(0, test.size()), random(0, maxTString));
  }
  // no erase(iter, ...)
  //   if (!test.empty()) {
  //     // TODO: is erase(end()) allowed?
  //     test.erase(test.begin() + random(0, test.size() - 1));
  //   }
  //   if (!test.empty()) {
  //     auto const i = test.begin() + random(0, test.size());
  //     if (i != test.end()) {
  //       test.erase(i, i + random(0, size_t(test.end() - i)));
  //     }
  //   }
  //   if (!test.empty()) {
  //     // Can't test pop_back with std::string, doesn't support it yet.
  //     // test.pop_back();
  //   }
}

template <class TString>
void clause11_21_4_6_6(TString& test) {
  auto pos = random(0, test.size());
  if (avoidAliasing) {
    test.replace(pos, random(0, test.size() - pos), TString(test));
  } else {
    test.replace(pos, random(0, test.size() - pos), test);
  }
  pos = random(0, test.size());
  TString s;
  randomTString(&s, maxTString);
  test.replace(pos, pos + random(0, test.size() - pos), s);
  auto pos1 = random(0, test.size());
  auto pos2 = random(0, test.size());
  if (avoidAliasing) {
    test.replace(pos1,
                 pos1 + random(0, test.size() - pos1),
                 TString(test),
                 pos2,
                 pos2 + random(0, test.size() - pos2));
  } else {
    test.replace(pos1,
                 pos1 + random(0, test.size() - pos1),
                 test,
                 pos2,
                 pos2 + random(0, test.size() - pos2));
  }
  pos1 = random(0, test.size());
  TString str;
  randomTString(&str, maxTString);
  pos2 = random(0, str.size());
  test.replace(
      pos1, pos1 + random(0, test.size() - pos1), str, pos2, pos2 + random(0, str.size() - pos2));
  pos = random(0, test.size());
  if (avoidAliasing) {
    test.replace(pos, random(0, test.size() - pos), TString(test).c_str(), test.size());
  } else {
    test.replace(pos, random(0, test.size() - pos), test.c_str(), test.size());
  }
  pos = random(0, test.size());
  randomTString(&str, maxTString);
  test.replace(pos, pos + random(0, test.size() - pos), str.c_str(), str.size());
  pos = random(0, test.size());
  randomTString(&str, maxTString);
  test.replace(pos, pos + random(0, test.size() - pos), str.c_str());
  //   pos = random(0, test.size());
  //   test.replace(pos, random(0, test.size() - pos), random(0, maxTString), random('a', 'z'));
  //     pos = random(0, test.size());
  //     if (avoidAliasing) {
  //       auto newTString = TString(test);
  //       test.replace(
  //           test.begin() + pos,
  //           test.begin() + pos + random(0, test.size() - pos),
  //           newTString);
  //     } else {
  //       test.replace(
  //           test.begin() + pos,
  //           test.begin() + pos + random(0, test.size() - pos),
  //           test);
  //     }
  //     pos = random(0, test.size());
  //     if (avoidAliasing) {
  //       auto newTString = TString(test);
  //       test.replace(
  //           test.begin() + pos,
  //           test.begin() + pos + random(0, test.size() - pos),
  //           newTString.c_str(),
  //           test.size() - random(0, test.size()));
  //     } else {
  //       test.replace(
  //           test.begin() + pos,
  //           test.begin() + pos + random(0, test.size() - pos),
  //           test.c_str(),
  //           test.size() - random(0, test.size()));
  //     }
  //     pos = random(0, test.size());
  //     auto const n = random(0, test.size() - pos);
  //     typename TString::iterator b = test.begin();
  //     TString str1;
  //     randomTString(&str1, maxTString);
  //     const TString& str3 = str1;
  //     const typename TString::value_type* ss = str3.c_str();
  //     test.replace(b + pos, b + pos + n, ss);
  //     pos = random(0, test.size());
  //     test.replace(
  //         test.begin() + pos,
  //         test.begin() + pos + random(0, test.size() - pos),
  //         random(0, maxTString),
  //         random('a', 'z'));
}

template <class TString>
void clause11_21_4_6_7(TString& test) {
  // no copy
  //   std::vector<typename TString::value_type> vec(random(0, maxTString));
  //   if (vec.empty()) {
  //     return;
  //   }
  //   test.copy(vec.data(), vec.size(), random(0, test.size()));
}

template <class TString>
void clause11_21_4_6_8(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  s = test;
}

template <class TString>
void clause11_21_4_7_1(TString& test) {
  // 21.3.6 string operations
  // exercise c_str() and data()
  assert(test.c_str() == test.data());
  // exercise get_allocator()
  //   TString s;
  //   randomTString(&s, maxTString);
  //   DCHECK(test.get_allocator() == s.get_allocator());
}

template <class TString>
void clause11_21_4_7_2_a(TString& test) {
  TString str = test.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_a1(TString& test) {
  TString str = TString(test).substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_a2(TString& test) {
  auto const& cTest = test;
  TString str = cTest.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_b(TString& test) {
  //   auto from = random(0, test.size());
  //   auto length = random(0, test.size() - from);
  //   TString str = test.substr(from, length);
  //   Num2TString(
  //       test,
  //       test.find(str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_2_b1(TString& test) {
  //   auto from = random(0, test.size());
  //   auto length = random(0, test.size() - from);
  //   TString str = TString(test).substr(from, length);
  //   Num2TString(
  //       test,
  //       test.find(str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_2_b2(TString& test) {
  //   auto from = random(0, test.size());
  //   auto length = random(0, test.size() - from);
  //   const auto& cTest = test;
  //   TString str = cTest.substr(from, length);
  //   Num2TString(
  //       test,
  //       test.find(str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_2_c(TString& test) {
  TString str = test.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_c1(TString& test) {
  TString str = TString(test).substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_c2(TString& test) {
  const auto& cTest = test;
  TString str = cTest.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.find(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_2_d(TString& test) {
  Num2TString(test, test.find(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_3_a(TString& test) {
  TString str = test.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.rfind(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_3_b(TString& test) {
  //   TString str = test.substr(random(0, test.size()), random(0, test.size()));
  //   Num2TString(
  //       test,
  //       test.rfind(str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_3_c(TString& test) {
  TString str = test.substr(random(0, test.size()), random(0, test.size()));
  Num2TString(test, test.rfind(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_3_d(TString& test) {
  Num2TString(test, test.rfind(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_4_a(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_first_of(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_4_b(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   Num2TString(
  //       test,
  //       test.find_first_of(
  //           str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_4_c(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_first_of(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_4_d(TString& test) {
  Num2TString(test, test.find_first_of(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_5_a(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_last_of(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_5_b(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   Num2TString(
  //       test,
  //       test.find_last_of(
  //           str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_5_c(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_last_of(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_5_d(TString& test) {
  Num2TString(test, test.find_last_of(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_6_a(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_first_not_of(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_6_b(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   Num2TString(
  //       test,
  //       test.find_first_not_of(
  //           str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_6_c(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_first_not_of(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_6_d(TString& test) {
  Num2TString(test, test.find_first_not_of(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_7_a(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_last_not_of(str, random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_7_b(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   Num2TString(
  //       test,
  //       test.find_last_not_of(
  //           str.c_str(), random(0, test.size()), random(0, str.size())));
}

template <class TString>
void clause11_21_4_7_7_c(TString& test) {
  TString str;
  randomTString(&str, maxTString);
  Num2TString(test, test.find_last_not_of(str.c_str(), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_7_d(TString& test) {
  Num2TString(test, test.find_last_not_of(random('a', 'z'), random(0, test.size())));
}

template <class TString>
void clause11_21_4_7_8(TString& test) {
  test = test.substr(random(0, test.size()), random(0, test.size()));
}

template <class TString>
void clause11_21_4_7_9_a(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  int tristate = test.compare(s);
  if (tristate > 0) {
    tristate = 1;
  } else if (tristate < 0) {
    tristate = 2;
  }
  Num2TString(test, tristate);
}

template <class TString>
void clause11_21_4_7_9_b(TString& test) {
  //   TString s;
  //   randomTString(&s, maxTString);
  //   int tristate =
  //       test.compare(random(0, test.size()), random(0, test.size()), s);
  //   if (tristate > 0) {
  //     tristate = 1;
  //   } else if (tristate < 0) {
  //     tristate = 2;
  //   }
  //   Num2TString(test, tristate);
}

template <class TString>
void clause11_21_4_7_9_c(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   int tristate = test.compare(
  //       random(0, test.size()),
  //       random(0, test.size()),
  //       str,
  //       random(0, str.size()),
  //       random(0, str.size()));
  //   if (tristate > 0) {
  //     tristate = 1;
  //   } else if (tristate < 0) {
  //     tristate = 2;
  //   }
  //   Num2TString(test, tristate);
}

template <class TString>
void clause11_21_4_7_9_d(TString& test) {
  //   TString s;
  //   randomTString(&s, maxTString);
  //   int tristate = test.compare(s.c_str());
  //   if (tristate > 0) {
  //     tristate = 1;
  //   } else if (tristate < 0) {
  //     tristate = 2;
  //   }
  //   Num2TString(test, tristate);
}

template <class TString>
void clause11_21_4_7_9_e(TString& test) {
  //   TString str;
  //   randomTString(&str, maxTString);
  //   int tristate = test.compare(
  //       random(0, test.size()),
  //       random(0, test.size()),
  //       str.c_str(),
  //       random(0, str.size()));
  //   if (tristate > 0) {
  //     tristate = 1;
  //   } else if (tristate < 0) {
  //     tristate = 2;
  //   }
  //   Num2TString(test, tristate);
}

template <class TString>
void clause11_21_4_8_1_a(TString& test) {
  TString s1;
  randomTString(&s1, maxTString);
  TString s2;
  randomTString(&s2, maxTString);
  test = s1 + s2;
}

template <class TString>
void clause11_21_4_8_1_b(TString& test) {
  TString s1;
  randomTString(&s1, maxTString);
  TString s2;
  randomTString(&s2, maxTString);
  test = std::move(s1) + s2;
}

template <class TString>
void clause11_21_4_8_1_c(TString& test) {
  TString s1;
  randomTString(&s1, maxTString);
  TString s2;
  randomTString(&s2, maxTString);
  test = s1 + std::move(s2);
}

template <class TString>
void clause11_21_4_8_1_d(TString& test) {
  TString s1;
  randomTString(&s1, maxTString);
  TString s2;
  randomTString(&s2, maxTString);
  test = std::move(s1) + std::move(s2);
}

template <class TString>
void clause11_21_4_8_1_e(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  TString s1;
  randomTString(&s1, maxTString);
  test = s.c_str() + s1;
}

template <class TString>
void clause11_21_4_8_1_f(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  TString s1;
  randomTString(&s1, maxTString);
  test = s.c_str() + std::move(s1);
}

template <class TString>
void clause11_21_4_8_1_g(TString& test) {
  //   TString s;
  //   randomTString(&s, maxTString);
  //   test = typename TString::value_type(random('a', 'z')) + s;
}

template <class TString>
void clause11_21_4_8_1_h(TString& test) {
  //   TString s;
  //   randomTString(&s, maxTString);
  //   test = typename TString::value_type(random('a', 'z')) + std::move(s);
}

template <class TString>
void clause11_21_4_8_1_i(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  TString s1;
  randomTString(&s1, maxTString);
  test = s + s1.c_str();
}

template <class TString>
void clause11_21_4_8_1_j(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  TString s1;
  randomTString(&s1, maxTString);
  test = std::move(s) + s1.c_str();
}

template <class TString>
void clause11_21_4_8_1_k(TString& test) {
  //   TString s;
  //   randomTString(&s, maxTString);
  //   test = s + typename TString::value_type(random('a', 'z'));
}

template <class TString>
void clause11_21_4_8_1_l(TString& test) {
  TString s;
  randomTString(&s, maxTString);
  TString s1;
  randomTString(&s1, maxTString);
  test = std::move(s) + s1.c_str();
}

// Numbering here is from C++11
template <class TString>
void clause11_21_4_8_9_a(TString& test) {
  //   std::basic_stringstream<typename TString::value_type> stst(test.c_str());
  //   TString str;
  //   while (stst) {
  //     stst >> str;
  //     test += str + test;
  //   }
}

TEST(StringUnicode, testAllClauses) {
  EXPECT_TRUE(1) << "Starting with seed: " << seed;
  std::string r;
  String c;
  unicode_string wr;
  Unicode wc;
  int count = 0;

  auto l = [&](const char* const clause,
               void (*f_std_string)(std::string&),
               void (*f_unicode_string)(unicode_string&),
               void (*f_string)(String&),
               void (*f_unicode)(Unicode&)) {
    do {
      randomTString(&r);
      c = r;
      EXPECT_EQ(c, r);
      auto localSeed = seed + count;

      rng = RandomT(localSeed);
      f_std_string(r);
      rng = RandomT(localSeed);
      f_string(c);
      EXPECT_EQ(r, c) << "Clause: " << clause << " with String"
                      << "\nLengths: " << r.size() << " vs. " << c.size() << "\nReference: '" << r
                      << "'"
                      << "\nActual:    '" << c << "'";

      wr = unicode_string(r.begin(), r.end());
      wc = Unicode(wr.c_str());
      EXPECT_EQ(wc, wr);

      rng = RandomT(localSeed);
      f_unicode_string(wr);
      rng = RandomT(localSeed);
      f_unicode(wc);
      EXPECT_EQ(wr, wc) << "Clause: " << clause << " with Unicode"
                        << "\nLengths: " << r.size() << " vs. " << c.size() << "\nReference: '" << r
                        << "'"
                        << "\nActual:    '" << c << "'";
    } while (++count % 100 != 0);
  };

#define TEST_CLAUSE(x)            \
  l(#x,                           \
    clause11_##x<std::string>,    \
    clause11_##x<unicode_string>, \
    clause11_##x<String>,         \
    clause11_##x<Unicode>);

  TEST_CLAUSE(21_4_2_a);
  TEST_CLAUSE(21_4_2_b);
  TEST_CLAUSE(21_4_2_c);
  TEST_CLAUSE(21_4_2_d);
  TEST_CLAUSE(21_4_2_e);
  TEST_CLAUSE(21_4_2_f);
  TEST_CLAUSE(21_4_2_g);
  TEST_CLAUSE(21_4_2_h);
  TEST_CLAUSE(21_4_2_i);
  TEST_CLAUSE(21_4_2_j);
  TEST_CLAUSE(21_4_2_k);
  TEST_CLAUSE(21_4_2_l);
  TEST_CLAUSE(21_4_2_lprime);
  TEST_CLAUSE(21_4_2_m);
  TEST_CLAUSE(21_4_2_n);
  TEST_CLAUSE(21_4_3);
  TEST_CLAUSE(21_4_4);
  TEST_CLAUSE(21_4_5);
  TEST_CLAUSE(21_4_6_1);
  TEST_CLAUSE(21_4_6_2);
  TEST_CLAUSE(21_4_6_3_a);
  TEST_CLAUSE(21_4_6_3_b);
  TEST_CLAUSE(21_4_6_3_c);
  TEST_CLAUSE(21_4_6_3_d);
  TEST_CLAUSE(21_4_6_3_e);
  TEST_CLAUSE(21_4_6_3_f);
  TEST_CLAUSE(21_4_6_3_g);
  TEST_CLAUSE(21_4_6_3_h);
  TEST_CLAUSE(21_4_6_3_i);
  TEST_CLAUSE(21_4_6_3_j);
  TEST_CLAUSE(21_4_6_3_k);
  TEST_CLAUSE(21_4_6_4);
  TEST_CLAUSE(21_4_6_5);
  TEST_CLAUSE(21_4_6_6);
  TEST_CLAUSE(21_4_6_7);
  TEST_CLAUSE(21_4_6_8);
  TEST_CLAUSE(21_4_7_1);

  TEST_CLAUSE(21_4_7_2_a);
  TEST_CLAUSE(21_4_7_2_a1);
  TEST_CLAUSE(21_4_7_2_a2);
  TEST_CLAUSE(21_4_7_2_b);
  TEST_CLAUSE(21_4_7_2_b1);
  TEST_CLAUSE(21_4_7_2_b2);
  TEST_CLAUSE(21_4_7_2_c);
  TEST_CLAUSE(21_4_7_2_c1);
  TEST_CLAUSE(21_4_7_2_c2);
  TEST_CLAUSE(21_4_7_2_d);
  TEST_CLAUSE(21_4_7_3_a);
  TEST_CLAUSE(21_4_7_3_b);
  TEST_CLAUSE(21_4_7_3_c);
  TEST_CLAUSE(21_4_7_3_d);
  TEST_CLAUSE(21_4_7_4_a);
  TEST_CLAUSE(21_4_7_4_b);
  TEST_CLAUSE(21_4_7_4_c);
  TEST_CLAUSE(21_4_7_4_d);
  TEST_CLAUSE(21_4_7_5_a);
  TEST_CLAUSE(21_4_7_5_b);
  TEST_CLAUSE(21_4_7_5_c);
  TEST_CLAUSE(21_4_7_5_d);
  TEST_CLAUSE(21_4_7_6_a);
  TEST_CLAUSE(21_4_7_6_b);
  TEST_CLAUSE(21_4_7_6_c);
  TEST_CLAUSE(21_4_7_6_d);
  TEST_CLAUSE(21_4_7_7_a);
  TEST_CLAUSE(21_4_7_7_b);
  TEST_CLAUSE(21_4_7_7_c);
  TEST_CLAUSE(21_4_7_7_d);
  TEST_CLAUSE(21_4_7_8);
  TEST_CLAUSE(21_4_7_9_a);
  TEST_CLAUSE(21_4_7_9_b);
  TEST_CLAUSE(21_4_7_9_c);
  TEST_CLAUSE(21_4_7_9_d);
  TEST_CLAUSE(21_4_7_9_e);
  TEST_CLAUSE(21_4_8_1_a);
  TEST_CLAUSE(21_4_8_1_b);
  TEST_CLAUSE(21_4_8_1_c);
  TEST_CLAUSE(21_4_8_1_d);
  TEST_CLAUSE(21_4_8_1_e);
  TEST_CLAUSE(21_4_8_1_f);
  TEST_CLAUSE(21_4_8_1_g);
  TEST_CLAUSE(21_4_8_1_h);
  TEST_CLAUSE(21_4_8_1_i);
  TEST_CLAUSE(21_4_8_1_j);
  TEST_CLAUSE(21_4_8_1_k);
  TEST_CLAUSE(21_4_8_1_l);
  TEST_CLAUSE(21_4_8_9_a);
}

// TEST(String, testGetline) {
//   std::string s1 =
//       "\
// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras accumsan \n\
// elit ut urna consectetur in sagittis mi auctor. Nulla facilisi. In nec \n\
// dolor leo, vitae imperdiet neque. Donec ut erat mauris, a faucibus \n\
// elit. Integer consectetur gravida augue, sit amet mattis mauris auctor \n\
// sed. Morbi congue libero eu nunc sodales adipiscing. In lectus nunc, \n\
// vulputate a fringilla at, venenatis quis justo. Proin eu velit \n\
// nibh. Maecenas vitae tellus eros. Pellentesque habitant morbi \n\
// tristique senectus et netus et malesuada fames ac turpis \n\
// egestas. Vivamus faucibus feugiat consequat. Donec fermentum neque sit \n\
// amet ligula suscipit porta. Phasellus facilisis felis in purus luctus \n\
// quis posuere leo tempor. Nam nunc purus, luctus a pharetra ut, \n\
// placerat at dui. Donec imperdiet, diam quis convallis pulvinar, dui \n\
// est commodo lorem, ut tincidunt diam nibh et nibh. Maecenas nec velit \n\
// massa, ut accumsan magna. Donec imperdiet tempor nisi et \n\
// laoreet. Phasellus lectus quam, ultricies ut tincidunt in, dignissim \n\
// id eros. Mauris vulputate tortor nec neque pellentesque sagittis quis \n\
// sed nisl. In diam lacus, lobortis ut posuere nec, ornare id quam.";

//   vector<String> v;
//   boost::split(v, s1, boost::is_any_of("\n"));
//   {
//     istringstream input(s1);
//     String line;
//     FOR_EACH (i, v) {
//       EXPECT_TRUE(!getline(input, line).fail());
//       EXPECT_EQ(line, *i);
//     }
//   }
// }

TEST(String, testMoveCtor) {
  // Move constructor. Make sure we allocate a large string, so the
  // small string optimization doesn't kick in.
  auto size = random(100, 2000);
  String s(size, 'a');
  String test = std::move(s);
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(size, test.size());
}

TEST(Unicode, testMoveCtor) {
  // Move constructor. Make sure we allocate a large string, so the
  // small string optimization doesn't kick in.
  auto size = random(100, 2000);
  Unicode s(size, 'a');
  Unicode test = std::move(s);
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(size, test.size());
}

TEST(String, testMoveAssign) {
  // Move constructor. Make sure we allocate a large string, so the
  // small string optimization doesn't kick in.
  auto size = random(100, 2000);
  String s(size, 'a');
  String test;
  test = std::move(s);
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(size, test.size());
}

TEST(Unicode, testMoveAssign) {
  // Move constructor. Make sure we allocate a large string, so the
  // small string optimization doesn't kick in.
  auto size = random(100, 2000);
  Unicode s(size, 'a');
  Unicode test;
  test = std::move(s);
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(size, test.size());
}

// TEST(String, testMoveOperatorPlusLhs) {
//   // Make sure we allocate a large string, so the
//   // small string optimization doesn't kick in.
//   auto size1 = random(100, 2000);
//   auto size2 = random(100, 2000);
//   String s1(size1, 'a');
//   String s2(size2, 'b');
//   String test;
//   test = std::move(s1) + s2;
//   EXPECT_TRUE(s1.empty());
//   EXPECT_EQ(size1 + size2, test.size());
// }

// TEST(Unicode, testMoveOperatorPlusLhs) {
//   // Make sure we allocate a large string, so the
//   // small string optimization doesn't kick in.
//   auto size1 = random(100, 2000);
//   auto size2 = random(100, 2000);
//   Unicode s1(size1, 'a');
//   Unicode s2(size2, 'b');
//   Unicode test;
//   test = std::move(s1) + s2;
//   debug(s1);
//   EXPECT_TRUE(s1.empty());
//   EXPECT_EQ(size1 + size2, test.size());
// }

// TEST(String, testMoveOperatorPlusRhs) {
//   // Make sure we allocate a large string, so the
//   // small string optimization doesn't kick in.
//   auto size1 = random(100, 2000);
//   auto size2 = random(100, 2000);
//   String s1(size1, 'a');
//   String s2(size2, 'b');
//   String test;
//   test = s1 + std::move(s2);
//   EXPECT_EQ(size1 + size2, test.size());
// }

// TEST(Unicode, testMoveOperatorPlusRhs) {
//   // Make sure we allocate a large unicode, so the
//   // small unicode optimization doesn't kick in.
//   auto size1 = random(100, 2000);
//   auto size2 = random(100, 2000);
//   Unicode s1(size1, 'a');
//   Unicode s2(size2, 'b');
//   Unicode test;
//   test = s1 + std::move(s2);
//   EXPECT_EQ(size1 + size2, test.size());
// }

// // The GNU C++ standard library throws an std::logic_error when an std::string
// // is constructed with a null pointer. Verify that we mirror this behavior.
// //
// // N.B. We behave this way even if the C++ library being used is something
// //      other than libstdc++. Someday if we deem it important to present
// //      identical undefined behavior for other platforms, we can re-visit this.
// TEST(String, testConstructionFromLiteralZero) {
//   EXPECT_THROW(String s(nullptr), std::logic_error);
// }

TEST(String, testFixedBugs_D479397) {
  String str(1337, 'f');
  String cp = str;
  cp.clear();
  cp.c_str();
  EXPECT_EQ(str.front(), 'f');
}

TEST(Unicode, testFixedBugs_D479397) {
  Unicode str(1337, 'f');
  Unicode cp = str;
  cp.clear();
  cp.c_str();
  EXPECT_EQ(str.front(), 'f');
}

TEST(String, testFixedBugs_D481173) {
  String str(1337, 'f');
  for (int i = 0; i < 2; ++i) {
    String cp = str;
    cp[1] = 'b';
    EXPECT_EQ(cp.c_str()[cp.size()], '\0');
    cp.push_back('?');
  }
}

TEST(Unicode, testFixedBugs_D481173) {
  Unicode str(1337, 'f');
  for (int i = 0; i < 2; ++i) {
    Unicode cp = str;
    cp[1] = 'b';
    EXPECT_EQ(cp.c_str()[cp.size()], '\0');
    cp.push_back('?');
  }
}

TEST(String, testFixedBugs_D580267_push_back) {
  String str(1337, 'f');
  String cp = str;
  cp.push_back('f');
}

TEST(Unicode, testFixedBugs_D580267_push_back) {
  Unicode str(1337, 'f');
  Unicode cp = str;
  cp.push_back('f');
}

TEST(String, testFixedBugs_D580267_operator_add_assign) {
  String str(1337, 'f');
  String cp = str;
  cp += "bb";
}

TEST(Unicode, testFixedBugs_D580267_operator_add_assign) {
  Unicode str(1337, 'f');
  Unicode cp = str;
  cp += U"bb";
}

TEST(String, testFixedBugs_D785057) {
  String str(1337, 'f');
  std::swap(str, str);
  EXPECT_EQ(1337, str.size());
}

TEST(Unicode, testFixedBugs_D785057) {
  Unicode str(1337, 'f');
  std::swap(str, str);
  EXPECT_EQ(1337, str.size());
}

TEST(String, testFixedBugs_D1012196_allocator_malloc) {
  String str(128, 'f');
  str.clear();       // Empty medium string.
  String copy(str);  // Medium string of 0 capacity.
  copy.push_back('b');
  EXPECT_GE(copy.capacity(), 1);
}

TEST(Unicode, testFixedBugs_D1012196_allocator_malloc) {
  Unicode str(128, 'f');
  str.clear();        // Empty medium unicode.
  Unicode copy(str);  // Medium unicode of 0 capacity.
  copy.push_back('b');
  EXPECT_GE(copy.capacity(), 1);
}

// no char + string
// TEST(String, testFixedBugs_D2813713) {
//   String s1("a");
//   s1.reserve(8); // Trigger the optimized code path.
//   auto test1 = '\0' + std::move(s1);
//   EXPECT_EQ(2, test1.size());

//   String s2(1, '\0');
//   s2.reserve(8);
//   auto test2 = "a" + std::move(s2);
//   EXPECT_EQ(2, test2.size());
// }
// TEST(Unicode, testFixedBugs_D2813713) {
//   Unicode s1("a");
//   s1.reserve(8); // Trigger the optimized code path.
//   auto test1 = '\0' + std::move(s1);
//   EXPECT_EQ(2, test1.size());

//   Unicode s2(1, '\0');
//   s2.reserve(8);
//   auto test2 = "a" + std::move(s2);
//   EXPECT_EQ(2, test2.size());
// }
TEST(String, testFixedBugs_D3698862) {
  EXPECT_EQ(String().find(String(), 4), String::npos);
}

TEST(Unicode, testFixedBugs_D3698862) {
  EXPECT_EQ(Unicode().find(Unicode(), 4), String::npos);
}

// TEST(String, testFixedBugs_D4355440) {
//   SKIP_IF(!usingJEMalloc());

//   String str(1337, 'f');
//   str.reserve(3840);
//   EXPECT_NE(str.capacity(), 3840);

//   struct DummyRefCounted {
//     std::atomic<size_t> refCount_;
//   };
//   EXPECT_EQ(
//       str.capacity(),
//       goodMallocSize(3840) - sizeof(DummyRefCounted) - sizeof(char));
// }

TEST(String, findWithNpos) {
  String fbstr("localhost:80");
  EXPECT_EQ(String::npos, fbstr.find(":", String::npos));
}

TEST(Unicode, findWithNpos) {
  Unicode fbstr(U"localhost:80");
  EXPECT_EQ(Unicode::npos, fbstr.find(U":", Unicode::npos));
}

TEST(String, testHash) {
  String a;
  String b;
  a.push_back(0);
  a.push_back(1);
  b.push_back(0);
  b.push_back(2);
  std::hash<String> hashfunc;
  EXPECT_NE(hashfunc(a), hashfunc(b));
}

TEST(Unicode, testHash) {
  Unicode a;
  Unicode b;
  a.push_back(0);
  a.push_back(1);
  b.push_back(0);
  b.push_back(2);
  std::hash<Unicode> hashfunc;
  EXPECT_NE(hashfunc(a), hashfunc(b));
}

TEST(String, testFrontBack) {
  String str("hello");
  EXPECT_EQ(str.front(), 'h');
  EXPECT_EQ(str.back(), 'o');
  str.front() = 'H';
  EXPECT_EQ(str.front(), 'H');
  str.back() = 'O';
  EXPECT_EQ(str.back(), 'O');
  EXPECT_EQ(str, "HellO");
}

TEST(Unicode, testFrontBack) {
  Unicode str(U"hello");
  EXPECT_EQ(str.front(), 'h');
  EXPECT_EQ(str.back(), 'o');
  str.front() = 'H';
  EXPECT_EQ(str.front(), 'H');
  str.back() = 'O';
  EXPECT_EQ(str.back(), 'O');
  EXPECT_EQ(str, U"HellO");
}

TEST(String, noexcept) {
  EXPECT_TRUE(noexcept(String()));
  String x;
  EXPECT_FALSE(noexcept(String(x)));
  EXPECT_TRUE(noexcept(String(std::move(x))));
  String y;
  EXPECT_FALSE(noexcept(y = x));
  EXPECT_TRUE(noexcept(y = std::move(x)));
}

TEST(Unicode, noexcept) {
  EXPECT_TRUE(noexcept(Unicode()));
  Unicode x;
  EXPECT_FALSE(noexcept(Unicode(x)));
  EXPECT_TRUE(noexcept(Unicode(std::move(x))));
  Unicode y;
  EXPECT_FALSE(noexcept(y = x));
  EXPECT_TRUE(noexcept(y = std::move(x)));
}

// TEST(String, iomanip) {
//   stringstream ss;
//   String fbstr("Hello");

//   ss << setw(6) << fbstr;
//   EXPECT_EQ(ss.str(), " Hello");
//   ss.str("");

//   ss << left << setw(6) << fbstr;
//   EXPECT_EQ(ss.str(), "Hello ");
//   ss.str("");

//   ss << right << setw(6) << fbstr;
//   EXPECT_EQ(ss.str(), " Hello");
//   ss.str("");

//   ss << setw(4) << fbstr;
//   EXPECT_EQ(ss.str(), "Hello");
//   ss.str("");

//   ss << setfill('^') << setw(6) << fbstr;
//   EXPECT_EQ(ss.str(), "^Hello");
//   ss.str("");
// }

// no replace(iter, iter, iter, iter)
// TEST(String, rvalueIterators) {
//   // you cannot take &* of a move-iterator, so use that for testing
//   String s = "base";
//   String r = "hello";
//   r.replace(
//       r.begin(),
//       r.end(),
//       std::make_move_iterator(s.begin()),
//       std::make_move_iterator(s.end()));
//   EXPECT_EQ("base", r);

//   // The following test is probably not required by the standard.
//   // i.e. this could be in the realm of undefined behavior.
//   String b = "123abcXYZ";
//   auto ait = b.begin() + 3;
//   auto Xit = b.begin() + 6;
//   b.replace(ait, b.end(), b.begin(), Xit);
//   EXPECT_EQ("123123abc", b); // if things go wrong, you'd get "123123123"
// }

TEST(String, moveTerminator) {
  // The source of a move must remain in a valid state
  String s(100, 'x');  // too big to be in-situ
  String k;
  k = std::move(s);

  EXPECT_EQ(0, s.size());
  EXPECT_EQ('\0', *s.c_str());
}

TEST(Unicode, moveTerminator) {
  // The source of a move must remain in a valid state
  Unicode s(100, 'x');  // too big to be in-situ
  Unicode k;
  k = std::move(s);

  EXPECT_EQ(0, s.size());
  EXPECT_EQ('\0', *s.c_str());
}

// namespace {
// /*
//  * t8968589: Clang 3.7 refused to compile w/ certain constructors (specifically
//  * those that were "explicit" and had a defaulted parameter, if they were used
//  * in structs which were default-initialized).  Exercise these just to ensure
//  * they compile.
//  *
//  * In diff D2632953 the old constructor:
//  *   explicit basic_String(const A& a = A()) noexcept;
//  *
//  * was split into these two, as a workaround:
//  *   basic_String() noexcept;
//  *   explicit basic_String(const A& a) noexcept;
//  */

struct TestStructDefaultAllocator {
  String stringMember;
  Unicode unicodeMember;
};

// std::atomic<size_t> allocatorConstructedCount(0);
// struct TestStructTStringAllocator : std::allocator<char> {
//   TestStructTStringAllocator() { ++allocatorConstructedCount; }
// };

// } // namespace

TEST(StringUnicodeCtorTest, DefaultInitStructDefaultAlloc) {
  TestStructDefaultAllocator t1{};
  EXPECT_TRUE(t1.stringMember.empty());
  EXPECT_TRUE(t1.unicodeMember.empty());
}

TEST(StringCtorTest, NullZeroConstruction) {
  String::value_type* p = nullptr;
  int n = 0;
  String f(p, n);
  EXPECT_EQ(f.size(), 0);
}

TEST(UnicodeCtorTest, NullZeroConstruction) {
  Unicode::value_type* p = nullptr;
  int n = 0;
  Unicode f(p, n);
  EXPECT_EQ(f.size(), 0);
}

// // Tests for the comparison operators. I use EXPECT_TRUE rather than EXPECT_LE
// // because what's under test is the operator rather than the relation between
// // the objects.

TEST(String, compareToStdTString) {
  using namespace std::string_literals;
  auto stdA = "a"s;
  auto stdB = "b"s;
  String fbA("a");
  String fbB("b");
  EXPECT_TRUE(stdA == fbA);
  EXPECT_TRUE(fbB == stdB);
  EXPECT_TRUE(stdA != fbB);
  EXPECT_TRUE(fbA != stdB);
  EXPECT_TRUE(stdA < fbB);
  EXPECT_TRUE(fbA < stdB);
  EXPECT_TRUE(stdB > fbA);
  EXPECT_TRUE(fbB > stdA);
  EXPECT_TRUE(stdA <= fbB);
  EXPECT_TRUE(fbA <= stdB);
  EXPECT_TRUE(stdA <= fbA);
  EXPECT_TRUE(fbA <= stdA);
  EXPECT_TRUE(stdB >= fbA);
  EXPECT_TRUE(fbB >= stdA);
  EXPECT_TRUE(stdB >= fbB);
  EXPECT_TRUE(fbB >= stdB);
}

TEST(Unicode, compareToStdU32TString) {
  using namespace std::string_literals;
  auto stdA = U"a"s;
  auto stdB = U"b"s;
  Unicode fbA(U"a");
  Unicode fbB(U"b");
  EXPECT_TRUE(stdA == fbA);
  EXPECT_TRUE(fbB == stdB);
  EXPECT_TRUE(stdA != fbB);
  EXPECT_TRUE(fbA != stdB);
  EXPECT_TRUE(stdA < fbB);
  EXPECT_TRUE(fbA < stdB);
  EXPECT_TRUE(stdB > fbA);
  EXPECT_TRUE(fbB > stdA);
  EXPECT_TRUE(stdA <= fbB);
  EXPECT_TRUE(fbA <= stdB);
  EXPECT_TRUE(stdA <= fbA);
  EXPECT_TRUE(fbA <= stdA);
  EXPECT_TRUE(stdB >= fbA);
  EXPECT_TRUE(fbB >= stdA);
  EXPECT_TRUE(stdB >= fbB);
  EXPECT_TRUE(fbB >= stdB);
}

// // Same again, but with a more challenging input - a common prefix and different
// // lengths.

TEST(String, compareToStdTStringLong) {
  using namespace std::string_literals;
  auto stdA = "1234567890a"s;
  auto stdB = "1234567890ab"s;
  String fbA("1234567890a");
  String fbB("1234567890ab");
  EXPECT_TRUE(stdA == fbA);
  EXPECT_TRUE(fbB == stdB);
  EXPECT_TRUE(stdA != fbB);
  EXPECT_TRUE(fbA != stdB);
  EXPECT_TRUE(stdA < fbB);
  EXPECT_TRUE(fbA < stdB);
  EXPECT_TRUE(stdB > fbA);
  EXPECT_TRUE(fbB > stdA);
  EXPECT_TRUE(stdA <= fbB);
  EXPECT_TRUE(fbA <= stdB);
  EXPECT_TRUE(stdA <= fbA);
  EXPECT_TRUE(fbA <= stdA);
  EXPECT_TRUE(stdB >= fbA);
  EXPECT_TRUE(fbB >= stdA);
  EXPECT_TRUE(stdB >= fbB);
  EXPECT_TRUE(fbB >= stdB);
}

TEST(Unicode, compareToStdU32TStringLong) {
  using namespace std::string_literals;
  auto stdA = U"1234567890a"s;
  auto stdB = U"1234567890ab"s;
  Unicode fbA(U"1234567890a");
  Unicode fbB(U"1234567890ab");
  EXPECT_TRUE(stdA == fbA);
  EXPECT_TRUE(fbB == stdB);
  EXPECT_TRUE(stdA != fbB);
  EXPECT_TRUE(fbA != stdB);
  EXPECT_TRUE(stdA < fbB);
  EXPECT_TRUE(fbA < stdB);
  EXPECT_TRUE(stdB > fbA);
  EXPECT_TRUE(fbB > stdA);
  EXPECT_TRUE(stdA <= fbB);
  EXPECT_TRUE(fbA <= stdB);
  EXPECT_TRUE(stdA <= fbA);
  EXPECT_TRUE(fbA <= stdA);
  EXPECT_TRUE(stdB >= fbA);
  EXPECT_TRUE(fbB >= stdA);
  EXPECT_TRUE(stdB >= fbB);
  EXPECT_TRUE(fbB >= stdB);
}

}  // namespace runtime
}  // namespace matxscript
