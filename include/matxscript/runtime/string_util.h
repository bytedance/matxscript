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

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace matxscript {
namespace runtime {
namespace StringUtil {

extern std::string Format(const char* fmt, ...);

extern bool Find(const std::string& source, const std::string& target);

extern int Find(const char* source, size_t source_len, const char* target, size_t target_len);

extern std::vector<std::string> Split(const std::string& input,
                                      const std::string& seps,
                                      bool preserve_all_tokens = true);

extern std::string Concat(const std::vector<std::string>& inputs, const std::string& sep = "");

extern std::string& Replace(std::string& input, const std::string& t, const std::string& r);

}  // namespace StringUtil
}  // namespace runtime
}  // namespace matxscript
