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
#include <matxscript/runtime/builtins_modules/_randommodule.h>

namespace matxscript {
namespace runtime {

TEST(random, random) {
  std::cout << "random list: [";
  for (int i = 0; i < 10; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << kernel_random_random();
  }
  std::cout << "]" << std::endl;
}

TEST(random, getstate) {
  std::cout << kernel_random_getstate() << std::endl;
}

TEST(random, seed) {
  kernel_random_seed();
  std::cout << kernel_random_getstate() << std::endl;
  kernel_random_seed({10});
  std::cout << kernel_random_getstate() << std::endl;
}

TEST(random, setstate) {
  auto state = kernel_random_getstate();
  kernel_random_setstate(state);
  std::cout << kernel_random_getstate() << std::endl;
}

TEST(random, getrandbits) {
  for (auto i = 1; i < 64; ++i) {
    std::cout << "getrandbits(" << i << "): " << kernel_random_getrandbits(i) << std::endl;
  }
}

TEST(random, uniform) {
  std::cout << "random.uniform list: [";
  for (int i = 0; i < 10; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << kernel_random_uniform(1, 2);
  }
  std::cout << "]" << std::endl;
}

TEST(random, triangular) {
  std::cout << "random.triangular: [";
  std::cout << kernel_random_triangular() << ", ";
  std::cout << kernel_random_triangular({}) << ", ";
  std::cout << kernel_random_triangular({0.1}) << ", ";
  std::cout << kernel_random_triangular({0.1, 0.9}) << ", ";
  std::cout << kernel_random_triangular({RTView(0.1), RTView(0.8), None.As<RTView>()});
  std::cout << "]" << std::endl;
}

TEST(random, randint) {
  std::cout << "random.randint: [";
  std::cout << kernel_random_randint(0, 10) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, normalvariate) {
  std::cout << "random.normalvariate: [";
  std::cout << kernel_random_normalvariate(0.0, 1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, lognormvariate) {
  std::cout << "random.lognormvariate: [";
  std::cout << kernel_random_lognormvariate(0.0, 1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, vonmisesvariate) {
  std::cout << "random.vonmisesvariate: [";
  std::cout << kernel_random_vonmisesvariate(0.0, 1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, gammavariate) {
  std::cout << "random.gammavariate: [";
  std::cout << kernel_random_gammavariate(0.01, 1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, gauss) {
  std::cout << "random.gauss: [";
  std::cout << kernel_random_gauss(0.01, 1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, betavariate) {
  std::cout << "random.betavariate: [";
  std::cout << kernel_random_betavariate(3.0, 3.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, expovariate) {
  std::cout << "random.expovariate: [";
  std::cout << kernel_random_expovariate(0.2) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, paretovariate) {
  std::cout << "random.paretovariate: [";
  std::cout << kernel_random_paretovariate(1.0) << ", ";
  std::cout << "]" << std::endl;
}

TEST(random, weibullvariate) {
  std::cout << "random.weibullvariate: [";
  std::cout << kernel_random_weibullvariate(0.1, 2.0) << ", ";
  std::cout << "]" << std::endl;
}

}  // namespace runtime
}  // namespace matxscript
