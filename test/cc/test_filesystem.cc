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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <ghc/filesystem.hpp>
#include <gtest/gtest.h>

#include "test_util.h"

namespace matxscript {
namespace runtime {

TEST(filesystem, du) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  fs::path dir{"."};

  uint64_t totalSize = 0;
  int totalDirs = 0;
  int totalFiles = 0;
  int maxDepth = 0;

  try {
    auto rdi = fs::recursive_directory_iterator(dir);
    for (auto de : rdi) {
      if (rdi.depth() > maxDepth) {
        maxDepth = rdi.depth();
      }
      if (de.is_regular_file()) {
        totalSize += de.file_size();
        ++totalFiles;
      } else if (de.is_directory()) {
        ++totalDirs;
      }
    }
  } catch (fs::filesystem_error fe) {
    std::cerr << "Error: " << fe.what() << std::endl;
    exit(1);
  }
  std::cout << totalSize << " bytes in " << totalFiles << " files and " << totalDirs
            << " directories, maximum depth: " << maxDepth << std::endl;
}

TEST(filesystem, dir) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  fs::path dir{"."};
  for (auto de : fs::directory_iterator(dir)) {
    auto ft = to_time_t(de.last_write_time());
    auto ftm = *std::localtime(&ft);
    std::cout << (de.is_directory() ? "d" : "-") << perm_to_str(de.symlink_status().permissions())
              << "  " << std::setw(8) << (de.is_directory() ? "-" : std::to_string(de.file_size()))
              << "  " << std::put_time(&ftm, "%Y-%m-%d %H:%M:%S") << "  "
              << de.path().filename().string() << std::endl;
  }
}

TEST(filesystem, copy_R_case1) {
  // source is a regular dir and subdir has symlink.
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;
  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");
  fs::create_directory("dir1/dir2");
  generateFile("dir1/dir2/file3");

  const auto copy_options = fs::copy_options::recursive;

  fs::copy("dir1", "dir3", copy_options);
  ASSERT_TRUE(!ec);
  ASSERT_TRUE(fs::exists("dir3/file1"));
  ASSERT_TRUE(fs::exists("dir3/file2"));
  ASSERT_TRUE(fs::is_regular_file("dir3/file2"));
  ASSERT_TRUE(fs::exists("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/dir2/file3"));
}

TEST(filesystem, copy_R_case2) {
  // source is a symlink and subdir has symlink.
  using namespace ::matxscript::test;
  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  fs::create_directory("test");
  generateFile("test/abc");

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");
  fs::create_directory("dir1/dir2");
  generateFile("dir1/dir2/file3");
  fs::create_symlink("dir1", "dir1_sym");
  ASSERT_TRUE(fs::exists("dir1_sym"));
  ASSERT_TRUE(fs::is_symlink("dir1_sym"));

  fs::create_symlink("../../test/abc", "dir1/dir2/abc_sym");
  ASSERT_TRUE(fs::exists("dir1/dir2/abc_sym"));
  ASSERT_TRUE(fs::is_symlink("dir1/dir2/abc_sym"));

  const auto copy_options = fs::copy_options::recursive;

  fs::copy("dir1_sym", "dir3", copy_options);
  ASSERT_TRUE(!ec);
  ASSERT_TRUE(fs::exists("dir3/file1"));
  ASSERT_TRUE(fs::exists("dir3/file2"));
  ASSERT_TRUE(fs::exists("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/file2"));
  ASSERT_TRUE(fs::is_regular_file("dir3/file1"));
  ASSERT_TRUE(fs::exists("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/dir2/file3"));
}

TEST(filesystem, copy_rd_case1) {
  using namespace ::matxscript::test;
  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");
  fs::create_directory("dir1/dir2");
  generateFile("dir1/dir2/file3");
  fs::create_symlink("dir1", "dir1_sym");
  // check src symlink
  ASSERT_TRUE(fs::exists("dir1_sym"));
  ASSERT_TRUE(fs::is_symlink("dir1_sym"));
  const auto copy_options = fs::copy_options::recursive | fs::copy_options::copy_symlinks;
  fs::copy("dir1_sym", "dir3", copy_options);
  ASSERT_TRUE(!ec);
  ASSERT_TRUE(fs::exists("dir3/file1"));
  ASSERT_TRUE(fs::is_symlink("dir3"));
  ASSERT_TRUE(fs::exists("dir3/file2"));
  ASSERT_TRUE(fs::is_regular_file("dir3/file2"));
  ASSERT_TRUE(fs::exists("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/dir2/file3"));
  ;
}

TEST(filesystem, copy_rd_case2) {
  using namespace ::matxscript::test;
  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  fs::create_directory("test");
  generateFile("test/abc");

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");
  fs::create_directory("dir1/dir2");
  generateFile("dir1/dir2/file3");

  fs::create_symlink("../../test/abc", "dir1/dir2/abc_sym");
  ASSERT_TRUE(fs::exists("dir1/dir2/abc_sym"));
  ASSERT_TRUE(fs::is_symlink("dir1/dir2/abc_sym"));

  const auto copy_options = fs::copy_options::recursive | fs::copy_options::copy_symlinks;
  fs::copy("dir1", "dir3", copy_options);
  ASSERT_TRUE(!ec);
  ASSERT_TRUE(fs::exists("dir3/file1"));
  ASSERT_TRUE(fs::is_directory("dir3"));
  ASSERT_TRUE(fs::exists("dir3/file2"));
  ASSERT_TRUE(fs::is_regular_file("dir3/file2"));
  ASSERT_TRUE(fs::exists("dir3/dir2/file3"));
  ASSERT_TRUE(fs::is_regular_file("dir3/dir2/file3"));
  ;
}

TEST(filesystem, create_directories) {
  using namespace ::matxscript::test;
  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directories("dir1/dir2/dir3/abc/", ec);
  ASSERT_TRUE(!ec);
  generateFile("dir1/file1");
  generateFile("dir1/file2");
  generateFile("dir1/dir2/file3");

  ASSERT_TRUE(fs::exists("dir1/dir2/dir3/abc"));
  ASSERT_TRUE(fs::is_directory("dir1/dir2/dir3"));
  ASSERT_TRUE(fs::exists("dir1/dir2/dir3/abc"));
}

}  // namespace runtime
}  // namespace matxscript
