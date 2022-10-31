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

#include <matxscript/runtime/container/file_ref.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

RTValue json_load(const File& fp);

RTValue json_loads(string_view s);

// TODO(wuxian): Need a file class that supports write operations
// RTValue json_dump(const RTValue& obj, const File& fp, int indent = -1, bool ensure_ascii = true);

Unicode json_dumps(const Any& obj, int indent = -1, bool ensure_ascii = true);

}  // namespace runtime
}  // namespace matxscript
