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
#include <fstream>
#include <iostream>
#include "matxscript/runtime/utf8_util.h"

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/file_ref.h>
#include <matxscript/runtime/jsonlib/json.h>

namespace matxscript {
namespace runtime {

TEST(Json, loads) {
  Unicode s(U"[1,2,\"3\"]");
  RTValue v = json_loads(UTF8Encode(s));
  std::cout << v << std::endl;
}

TEST(Json, load) {
  Unicode fn(U"./json_load_data.json");
  std::ofstream of(fn.encode());
  of << "{\"1\": \"23\", \"4\": [5, 6]}";
  of.close();
  File fp(fn);
  RTValue v = json_load(fp);
  std::cout << v << std::endl;
}

TEST(Json, dumps) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}, {3, Unicode(U"matx4")}};
  Unicode s = json_dumps(RTView(d));
  std::cout << s << std::endl;
  s = json_dumps(RTView(d), 2);
  std::cout << s << std::endl;
  s = json_dumps(RTView(d), 2, false);
  std::cout << s << std::endl;
}

}  // namespace runtime
}  // namespace matxscript
