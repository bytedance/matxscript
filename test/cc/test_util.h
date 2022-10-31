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
#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <ghc/filesystem.hpp>
#include <gtest/gtest.h>

namespace matxscript {
namespace test {

namespace fs = ghc::filesystem;
enum class TempOpt { none, change_path };
class TemporaryDirectory {
 public:
  TemporaryDirectory(TempOpt opt = TempOpt::none) {
    static auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<unsigned int>(seed) ^
                     static_cast<unsigned int>(reinterpret_cast<ptrdiff_t>(&opt)));
    std::uniform_int_distribution<int> random_dist(0, 35);  // define the range
    do {
      std::string filename;
      filename.reserve(16);
      filename.append("test_");
      for (int i = 0; i < 8; ++i) {
        filename.push_back("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[random_dist(gen)]);
      }
      tmp_path_ = fs::canonical(fs::temp_directory_path()) / filename;
    } while (fs::exists(tmp_path_));

    fs::create_directories(tmp_path_);
    if (opt == TempOpt::change_path) {
      orig_dir_ = fs::current_path();
      fs::current_path(tmp_path_);
    }
  }

  ~TemporaryDirectory() {
    // restore current_path
    if (!orig_dir_.empty()) {
      fs::current_path(orig_dir_);
    }
    // clear tmp path
    fs::remove_all(tmp_path_);
  }

  const fs::path& path() const {
    return tmp_path_;
  }

 private:
  fs::path tmp_path_;
  fs::path orig_dir_;
};

static void generateFile(const fs::path& pathname, int withSize = -1) {
  fs::ofstream outfile(pathname);
  if (withSize < 0) {
    outfile << "Hello world!" << std::endl;
  } else {
    outfile << std::string(size_t(withSize), '*');
  }
}

template <typename TP>
std::time_t to_time_t(TP tp) {
  // Based on trick from: Nico Josuttis, C++17 - The Complete Guide
  std::chrono::system_clock::duration dt =
      std::chrono::duration_cast<std::chrono::system_clock::duration>(tp - TP::clock::now());
  return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() + dt);
}

static std::string perm_to_str(fs::perms prms) {
  std::string result;
  result.reserve(9);
  for (int i = 0; i < 9; ++i) {
    result = ((static_cast<int>(prms) & (1 << i)) ? "xwrxwrxwr"[i] : '-') + result;
  }

  return result;
}
}  // namespace test
}  // namespace matxscript
