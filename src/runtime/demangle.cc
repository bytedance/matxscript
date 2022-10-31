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
#include <cxxabi.h>

#include <matxscript/runtime/demangle.h>

namespace matxscript {
namespace runtime {

String Demangle(const char* mangled) {
  String msg(mangled);
  size_t symbol_start = String::npos;
  size_t symbol_end = String::npos;
  if (((symbol_start = msg.find("_Z")) != String::npos) &&
      (symbol_end = msg.find_first_of(" +", symbol_start))) {
    String left_of_symbol(msg, 0, symbol_start);
    String symbol(msg, symbol_start, symbol_end - symbol_start);
    String right_of_symbol(msg, symbol_end);

    int status = 0;
    size_t length = symbol.size();
    std::unique_ptr<char, void (*)(void* __ptr)> demangled_symbol = {
        abi::__cxa_demangle(symbol.c_str(), nullptr, &length, &status), &std::free};
    if (demangled_symbol && status == 0 && length > 0) {
      return left_of_symbol + String(demangled_symbol.get()) + right_of_symbol;
    }
  }
  return String(mangled);
}

String DemangleType(const char* mangled) {
  String symbol(mangled);
  int status = 0;
  size_t length = symbol.size();
  std::unique_ptr<char, void (*)(void* __ptr)> demangled_symbol = {
      abi::__cxa_demangle(symbol.c_str(), nullptr, &length, &status), &std::free};
  if (demangled_symbol && status == 0 && length > 0) {
    return String(demangled_symbol.get());
  }
  return String(mangled);
}

}  // namespace runtime
}  // namespace matxscript
