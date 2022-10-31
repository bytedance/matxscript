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
#include <matxscript/runtime/file_reader.h>
#include <cstdio>
#include <fstream>

namespace matxscript {
namespace runtime {

class FileTest : public ::testing::Test {
 protected:
  FileTest() {
  }

  virtual ~FileTest() {
  }

  virtual void SetUp() {
    std::ofstream fout("./input.utf8.txt");
    fout << "line 1" << std::endl;
    fout << "\u884c 2" << std::endl;
    fout << "\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631" << std::endl;
    fout.close();

    fout.open("./input.utf8.txt.nonewline");
    fout << "line 1" << std::endl;
    fout << "\u884c 2" << std::endl;
    fout << "\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631";
    fout.close();

    fout.open("./input.large_line.txt");
    fout << "sep" << std::endl;
    std::string data;
    data.resize(buffer_size - 1, 'A');
    fout << data << std::endl;
    fout << "sep" << std::endl;
    data.resize(buffer_size, 'A');
    fout << data << std::endl;
    fout << "sep" << std::endl;
    data.resize(buffer_size + 1, 'A');
    fout << data << std::endl;
    fout << "sep" << std::endl;
    fout.close();
  }

  virtual void TearDown() {
    remove("./input.utf8.txt");
    remove("./input.utf8.txt.nonewline");
    remove("./input.large_line.txt");
  }

  static constexpr auto buffer_size = 8 * 1024 * 1024;
};

TEST_F(FileTest, FileReader) {
  FileReader reader("./input.utf8.txt", true);
  const char* line = nullptr;
  size_t len = 0;
  ASSERT_TRUE(reader.ReadLine(&line, &len));
  ASSERT_EQ(String(line, len), String("line 1\n"));
  ASSERT_TRUE(reader.ReadLine(&line, &len));
  ASSERT_EQ(String(line, len), String("\u884c 2\n"));
  ASSERT_TRUE(reader.ReadLine(&line, &len));
  ASSERT_EQ(String(line, len),
            String("\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631\n"));
  ASSERT_FALSE(reader.ReadLine(&line, &len));
  ASSERT_EQ(String(line, len), String(""));
}

TEST_F(FileTest, FileReaderLargeLine) {
  const char* line = nullptr;
  size_t len = 0;
  {
    std::string sep("sep\n");
    std::string data1;
    data1.resize(buffer_size - 1, 'A');
    data1.push_back('\n');
    std::string data2;
    data2.resize(buffer_size, 'A');
    data2.push_back('\n');
    std::string data3;
    data3.resize(buffer_size + 1, 'A');
    data3.push_back('\n');

    FileReader reader("./input.large_line.txt", true);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data1);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data2);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data3);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
  }
  {
    std::string sep("sep");
    std::string data1;
    data1.resize(buffer_size - 1, 'A');
    std::string data2;
    data2.resize(buffer_size, 'A');
    std::string data3;
    data3.resize(buffer_size + 1, 'A');
    FileReader reader("./input.large_line.txt", false);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data1);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data2);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), data3);

    ASSERT_TRUE(reader.ReadLine(&line, &len));
    ASSERT_EQ(std::string(line, len), sep);
  }
}

TEST_F(FileTest, readline_string) {
  File f(U"./input.utf8.txt");
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String("line 1\n"));
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String("\u884c 2\n"));
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(),
            String("\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631\n"));
  ASSERT_FALSE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String(""));
}

TEST_F(FileTest, readline_string_nonewline) {
  File f(U"./input.utf8.txt.nonewline");
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String("line 1\n"));
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String("\u884c 2\n"));
  ASSERT_TRUE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(),
            String("\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631"));
  ASSERT_FALSE(f.HasNext());
  ASSERT_EQ(f.ReadLineString(), String(""));
}

TEST_F(FileTest, readline_unicode) {
  Unicode uline;
  File uf(U"./input.utf8.txt", U"rb");
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U"line 1\n"));
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U"\u884c 2\n"));
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(),
            Unicode(U"\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631\n"));
  ASSERT_FALSE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U""));
}

TEST_F(FileTest, readline_unicode_nonewline) {
  Unicode uline;
  File uf(U"./input.utf8.txt.nonewline", U"rb");
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U"line 1\n"));
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U"\u884c 2\n"));
  ASSERT_TRUE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(),
            Unicode(U"\u062b\u0644\u0627\u062b\u0629\u0020\u0623\u0633\u0637\u0631"));
  ASSERT_FALSE(uf.HasNext());
  ASSERT_EQ(uf.ReadLineUnicode(), Unicode(U""));
}

}  // namespace runtime
}  // namespace matxscript
