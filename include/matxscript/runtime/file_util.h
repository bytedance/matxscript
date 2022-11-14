// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
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
 * \file file_utils.h
 * \brief Minimum file manipulation utils for runtime.
 */
#pragma once

#include <limits.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <system_error>
#include <unordered_map>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/string_util.h>

namespace matxscript {
namespace runtime {
namespace FileUtil {
/*!
 * \brief Get file format from given file name or format argument.
 * \param file_name The name of the file.
 * \param format The format of the file.
 */
std::string GetFileFormat(string_view file_name, string_view format);

/*!
 * \brief Get meta file path given file name and format.
 * \param file_name The name of the file.
 */
std::string GetMetaFilePath(string_view file_name);

/*!
 * \brief Get file basename (i.e. without leading directories)
 * \param file_name The name of the file.
 * \return the base name
 */
std::string GetFileBasename(string_view file_name);

/*!
 * \brief Get file directory
 * \param file_name The name of the file.
 * \return the dir
 */
std::string GetFileDirectory(string_view file_name);

/*!
 * \brief Get file extension
 * \param file_name The name of the file.
 * \return the dir
 */
std::string GetFileExtension(string_view file_name);

/*!
 * \brief Load binary file into a in-memory buffer.
 * \param file_name The name of the file.
 * \param data The data to be loaded.
 */
void LoadBinaryFromFile(string_view file_name, std::string* data);

/*!
 * \brief Load binary file into a in-memory buffer.
 * \param file_name The name of the file.
 * \param data The binary data to be saved.
 */
void SaveBinaryToFile(string_view file_name, string_view data);

/*!
 * \brief Remove (unlink) a file.
 * \param file_name The file name.
 */
void RemoveFile(string_view file_name);

bool Exists(string_view name);

std::string RTrim(string_view input, string_view tr);

std::string BaseName(string_view location);

bool DirExists(string_view folder);

bool IsLinkDir(string_view folder);

bool IsRegularFile(string_view loc);

void Mkdir(string_view dir);

int Copy(string_view src, string_view dest);

}  // namespace FileUtil
}  // namespace runtime
}  // namespace matxscript
