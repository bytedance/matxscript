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
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/server/simple_mpmc_server.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace matxscript {
namespace runtime {

TEST(TXSession, Server) {
  // test case
  std::unordered_map<std::string, RTValue> feed_dict;
  feed_dict.emplace("text", String("this is a user query"));
  std::vector<std::pair<std::string, RTValue>> result;
  const char* module_path = "./TXSession_test_load_store";
  const char* module_name = "model.spec.json";

  {
    // create matx module for test
    TXSession sess;
    auto sym = sess.CreateVariable("text", String("test query"));
    sess.Trace(sym.get());
    result = sess.Run(feed_dict);
    for (auto& r : result) {
      std::cout << "key: " << r.first << ", value: " << r.second << std::endl;
    }
    sess.Save(module_path, module_name);
  }
  {
    auto sess = TXSession::Load(module_path, module_name);
    result = sess->Run(feed_dict);
    for (auto& r : result) {
      std::cout << "key: " << r.first << ", value: " << r.second << std::endl;
    }
  }
  {
    auto sess = TXSession::Load(module_path, module_name);
    std::vector<std::shared_ptr<TXSession>> handlers;
    handlers.push_back(std::move(sess));
    server::SimpleMPMCServer server(handlers, "TXSession");
    server.start();
    result = server.process(feed_dict);
    for (auto& r : result) {
      std::cout << "key: " << r.first << ", value: " << r.second << std::endl;
    }
    uint64_t enqueue_tc = 0;
    uint64_t run_tc = 0;
    result = server.process_with_tc(feed_dict, &enqueue_tc, &run_tc);
    for (auto& r : result) {
      std::cout << "key: " << r.first << ", value: " << r.second << std::endl;
    }
    std::cout << "enqueue_tc: " << enqueue_tc << "(us), run_tc: " << run_tc << "(us)" << std::endl;
    server.stop();
  }
}

}  // namespace runtime
}  // namespace matxscript
