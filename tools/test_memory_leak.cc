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
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/file_reader.h>
#include <matxscript/runtime/ft_container.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ::matxscript::runtime;

namespace {

void test_temp_dict(char* file_path, int repeat) {
  Dict cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.set_item(index++, String(line, len));
    }
  }
  std::cout << "dict size: " << cons.size() << std::endl;
}

void test_temp_set(char* file_path, int repeat) {
  Set cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.add(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "set size: " << cons.size() << std::endl;
}

void test_temp_list(char* file_path, int repeat) {
  List cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.append(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "list size: " << cons.size() << std::endl;
}

void test_temp_ft_dict(char* file_path, int repeat) {
  FTDict<int64_t, String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.set_item(index++, String(line, len));
    }
  }
  std::cout << "ft_dict size: " << cons.size() << std::endl;
}

void test_temp_ft_set(char* file_path, int repeat) {
  FTSet<String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.add(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "ft_set size: " << cons.size() << std::endl;
}

void test_temp_ft_list(char* file_path, int repeat) {
  FTList<String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.append(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "ft_list size: " << cons.size() << std::endl;
}

void test_temp_stl_dict(char* file_path, int repeat) {
  std::unordered_map<int64_t, String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.emplace(index++, String(line, len));
    }
  }
  std::cout << "ft_dict size: " << cons.size() << std::endl;
}

void test_temp_stl_set(char* file_path, int repeat) {
  std::unordered_set<String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.emplace(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "ft_set size: " << cons.size() << std::endl;
}

void test_temp_stl_list(char* file_path, int repeat) {
  std::vector<String> cons;
  FileReader reader(file_path);
  const char* line = nullptr;
  size_t len = 0;
  int64_t index = 0;
  while (reader.ReadLine(&line, &len)) {
    for (int i = 0; i < repeat; ++i) {
      cons.push_back(string_view(std::to_string(index++)) + String(line, len));
    }
  }
  std::cout << "ft_list size: " << cons.size() << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "arg num error" << std::endl;
    return -1;
  }
  char* file_path = argv[1];
  int repeat = std::stoi(argv[2]);

  std::cout << "begin load data" << std::endl;
  {
    test_temp_dict(file_path, repeat);
    test_temp_set(file_path, repeat);
    test_temp_list(file_path, repeat);
    test_temp_ft_dict(file_path, repeat);
    test_temp_ft_set(file_path, repeat);
    test_temp_ft_list(file_path, repeat);
    test_temp_stl_dict(file_path, repeat);
    test_temp_stl_set(file_path, repeat);
    test_temp_stl_list(file_path, repeat);
  }
  std::cout << "finish load data" << std::endl;
  sleep(120);
  return 0;
}
