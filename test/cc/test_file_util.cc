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
#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>

#include <ghc/filesystem.hpp>
#include <gtest/gtest.h>
#include <matxscript/runtime/file_util.h>

#include "test_util.h"

namespace matxscript {
namespace runtime {

TEST(FileUtil, GetFileFormat) {
  std::string test_path_str = "/tmp/foo.txt";
  std::string res1 = FileUtil::GetFileFormat(test_path_str, {""});
  std::cout << "[FileUtil::GetFileFormat] res1: " << res1 << std::endl;
  ASSERT_STREQ(res1.c_str(), "txt");

  std::string test_format_str = "test";
  std::string res2 = FileUtil::GetFileFormat(test_path_str, test_format_str);
  std::cout << "[FileUtil::GetFileFormat] res2: " << res2 << std::endl;
  ASSERT_STREQ(res2.c_str(), test_format_str.c_str());
}

TEST(FileUtil, GetFileBasename) {
  std::string test_path_str1 = "/tmp/foo.txt";
  std::string test_path_str2 = "foo.txt";
  std::string gt = "foo.txt";
  std::string res1 = FileUtil::GetFileBasename(test_path_str1);
  std::cout << "[FileUtil::GetFileBasename] res1: " << res1 << std::endl;
  ASSERT_STREQ(res1.c_str(), gt.c_str());

  std::string res2 = FileUtil::GetFileBasename(test_path_str2);
  std::cout << "[FileUtil::GetFileBasename] res2: " << res2 << std::endl;
  ASSERT_STREQ(res2.c_str(), gt.c_str());
}

TEST(FileUtil, BaseName) {
  std::string test_path_str1 = "//tmp///foo.txt";
  std::string test_path_str2 = "foo.txt";
  std::string gt = "foo.txt";
  std::string res1 = FileUtil::BaseName(test_path_str1);
  std::cout << "[FileUtil::BaseName] res1: " << res1 << std::endl;
  ASSERT_STREQ(res1.c_str(), gt.c_str());

  std::string res2 = FileUtil::BaseName(test_path_str2);
  std::cout << "[FileUtil::BaseName] res2: " << res2 << std::endl;
  ASSERT_STREQ(res2.c_str(), gt.c_str());
}

TEST(Copy, copy_regular_file) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");

  fs::create_directory("dir2");

  int ret = 1;
  ret = ::matxscript::runtime::FileUtil::Copy("dir1/file1", "dir2");
  ASSERT_EQ(ret, 0);

  ret = ::matxscript::runtime::FileUtil::Copy("dir1/file2", "dir2");
  ASSERT_EQ(ret, 0);

  ASSERT_TRUE(fs::exists("dir2/file1"));
  ASSERT_TRUE(fs::exists("dir2/file2"));

  ASSERT_TRUE(fs::is_regular_file("dir2/file1"));
  ASSERT_TRUE(fs::is_regular_file("dir2/file2"));
}

TEST(Copy, copy_symlink_file) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");

  fs::create_symlink("../dir1/file1", "dir1/file1_sym");
  fs::create_symlink("../dir1/file2", "dir1/file2_sym");

  fs::create_directory("dir2");
  // !!!NOTICE!!!
  // DeprecatedCopy can't handle symlink file properly.
  int ret = 1;
  ret = ::matxscript::runtime::FileUtil::Copy("dir1/file1_sym", "dir2");
  ASSERT_EQ(ret, 0);

  ret = ::matxscript::runtime::FileUtil::Copy("dir1/file2_sym", "dir2");
  ASSERT_EQ(ret, 0);
}

TEST(Copy, copy_regular_dir) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");

  fs::create_directory("dir2");

  int ret = 1;
  ret = ::matxscript::runtime::FileUtil::Copy("dir1", "dir2");
  ASSERT_EQ(ret, 0);

  ASSERT_TRUE(fs::exists("dir2/dir1/file1"));
  ASSERT_TRUE(fs::exists("dir2/dir1/file2"));

  ASSERT_TRUE(fs::is_regular_file("dir2/dir1/file1"));
  ASSERT_TRUE(fs::is_regular_file("dir2/dir1/file2"));
}

TEST(Copy, copy_regular_dir_already_exists) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");

  fs::create_directory("dir2");
  fs::create_directory("dir2/dir1");

  int ret = 1;
  ret = ::matxscript::runtime::FileUtil::Copy("dir1", "dir2");
  ASSERT_EQ(ret, 0);

  ASSERT_TRUE(fs::exists("dir2/dir1/file1"));
  ASSERT_TRUE(fs::exists("dir2/dir1/file2"));

  ASSERT_TRUE(fs::is_regular_file("dir2/dir1/file1"));
  ASSERT_TRUE(fs::is_regular_file("dir2/dir1/file2"));
}

TEST(Copy, copy_symlink_dir) {
  namespace fs = ghc::filesystem;
  using namespace ::matxscript::test;

  TemporaryDirectory t(TempOpt::change_path);
  std::cout << t.path() << std::endl;

  std::error_code ec;
  fs::create_directory("dir1");
  generateFile("dir1/file1");
  generateFile("dir1/file2");

  fs::create_symlink("dir1", "dir1_sym");
  fs::create_directory("dir2");

  int ret = 1;
  ret = ::matxscript::runtime::FileUtil::Copy("dir1_sym", "dir2");
  ASSERT_EQ(ret, 0);

  ASSERT_TRUE(fs::exists("dir2/dir1_sym/file1"));
  ASSERT_TRUE(fs::exists("dir2/dir1_sym/file2"));

  ASSERT_TRUE(fs::is_regular_file("dir2/dir1_sym/file1"));
  ASSERT_TRUE(fs::is_regular_file("dir2/dir1_sym/file2"));
}

}  // namespace runtime
}  // namespace matxscript
