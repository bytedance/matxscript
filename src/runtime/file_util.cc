// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
 *
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

/*!
 * \file file_utils.cc
 */

#include <matxscript/runtime/file_util.h>
#if defined(__unix__) || defined(__linux__) || defined(__gnu_linux__)
#include <sys/stat.h>
#endif

#include <fstream>
#include <unordered_map>

#include <ghc/filesystem.hpp>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {
namespace FileUtil {

std::string GetFileFormat(string_view file_name, string_view format) {
  if (format.length() == 0) {
    size_t pos = file_name.find_last_of(".");
    if (pos != std::string::npos) {
      return std::string(file_name.substr(pos + 1, file_name.length() - pos - 1));
    } else {
      return "";
    }
  } else {
    return std::string(format);
  }
}

std::string GetFileBasename(string_view file_name) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(file_name)};
  return p.filename();
}

std::string GetFileDirectory(string_view file_name) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(file_name)};
  return p.parent_path();
}

std::string GetMetaFilePath(string_view file_name) {
  size_t pos = file_name.find_last_of(".");
  if (pos != std::string::npos) {
    return std::string(file_name.substr(0, pos)) + ".matx_meta.json";
  } else {
    return std::string(file_name) + ".matx_meta.json";
  }
}

void LoadBinaryFromFile(string_view file_name, std::string* data) {
  std::ifstream fs(std::string(file_name), std::ios::in | std::ios::binary);
  MXCHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

void SaveBinaryToFile(string_view file_name, string_view data) {
  std::ofstream fs(std::string(file_name), std::ios::out | std::ios::binary);
  MXCHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.size());
}

void RemoveFile(string_view file_name) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(file_name)};
  fs::remove(p);
}

bool Exists(string_view name) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(name)};

  return fs::exists(p);
}

std::string RTrim(string_view input, string_view tr) {
  if (input.empty()) {
    return "";
  }
  int64_t end = input.length() - 1;
  for (; end >= 0; --end) {
    char c = input.at(end);
    if (tr.find(c) == std::string::npos) {
      break;
    }
  }
  return std::string(input.substr(0, end + 1));
}

std::string BaseName(string_view location) {
  std::string loc = RTrim(location, "/\\");
  namespace fs = ghc::filesystem;
  fs::path p{std::string(loc)};
  return p.filename();
}

bool DirExists(string_view folder) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(folder)};
  return fs::exists(p) && fs::is_directory(p);
}

bool IsLinkDir(string_view folder) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(folder)};
  return fs::exists(p) && fs::is_symlink(p);
}

bool IsRegularFile(string_view loc) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(loc)};
  return fs::exists(p) && fs::is_regular_file(p);
}

void Mkdir(string_view dir) {
  namespace fs = ghc::filesystem;
  fs::path p{std::string(dir)};
  try {
    fs::create_directories(p);
  } catch (const fs::filesystem_error& e) {
    MXLOG(FATAL) << e.what();
    std::cerr << e.what() << std::endl;
  }
}

int Copy(string_view src, string_view dest) {
  namespace fs = ghc::filesystem;

  fs::path src_path{std::string{src}};
  fs::path dest_path{std::string{dest}};

  if (!fs::exists(src_path) || !fs::exists(dest_path)) {
    MXLOG(FATAL) << "[Bundle][src:" << src << "][dst:" << dest << "] input src or dest is null";
    return -1;
  }
  MXLOG(INFO) << "[Bundle][src:" << src << "][is_link:" << IsLinkDir(src) << "][dst:" << dest
              << "] wait...";

  // handle regular_file and symlink_file
  if (fs::is_regular_file(src_path)) {
    try {
      const auto copy_options = fs::copy_options::recursive | fs::copy_options::update_existing;
      fs::copy(src_path, dest_path, copy_options);
    } catch (const fs::filesystem_error& e) {
      std::cerr << e.what() << std::endl;
      MXLOG(FATAL) << e.what();
      return -1;
    }
  }

  // handle directory and symlink directory
  if (fs::is_directory(src_path)) {
    std::string src_path_base_name = BaseName(src);
    try {
      // eg. src: dir1 && tgt: dir2
      // res: dir2/dir1
      dest_path.append(src_path_base_name);
      fs::create_directory(dest_path);

      const auto copy_options = fs::copy_options::recursive | fs::copy_options::update_existing;
      fs::copy(src_path, dest_path, copy_options);
    } catch (const fs::filesystem_error& e) {
      std::cerr << e.what() << std::endl;
      MXLOG(FATAL) << e.what();
      return -1;
    }
  }

  return 0;
}

}  // namespace FileUtil
}  // namespace runtime
}  // namespace matxscript
