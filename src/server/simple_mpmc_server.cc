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
#include <matxscript/server/simple_mpmc_server.h>

namespace matxscript {
namespace runtime {
namespace server {

void SimpleMPMCServer::ThreadEntry(SimpleMPMCServer* pool,
                                   TXSession* sess_ptr,
                                   const std::string& name) {
#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
  for (;;) {
    RunnablePtr task = nullptr;
    for (;;) {
      if (pool->tasks_.try_dequeue(task)) {
        break;
      }
      if (pool->stop_) {
        break;
      }
      // queue is full, sleep 1us
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
    if (pool->stop_) {
      return;
    } else {
      if (task != nullptr) {
        task->Run(sess_ptr);
        std::atomic_thread_fence(std::memory_order_acquire);
        task->SetDone();
      }
    }
  }
}

SimpleMPMCServer::SimpleMPMCServer(std::vector<std::shared_ptr<TXSession>> handlers,
                                   std::string name)
    : stop_(false), name_(std::move(name)), tasks_(1024), handlers_(std::move(handlers)) {
}

void SimpleMPMCServer::start() {
  for (size_t i = 0; i < handlers_.size(); ++i) {
    char buffer[16] = {0};
    snprintf(buffer, sizeof(buffer), "T%zu.%s", i, name_.c_str());
    workers_.emplace_back(
        SimpleMPMCServer::ThreadEntry, this, handlers_[i].get(), std::string(buffer));
  }
}

void SimpleMPMCServer::stop() {
  stop_ = true;
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

std::vector<std::pair<std::string, RTValue>> SimpleMPMCServer::process(
    const std::unordered_map<std::string, RTValue>& feed_dict) {
  std::vector<std::pair<std::string, RTValue>> outputs;
  auto runner = std::make_shared<Runnable>(&feed_dict, &outputs);
  // push task
  for (;;) {
    if (tasks_.try_enqueue(runner)) {
      break;
    }
    // queue is full, sleep 1us
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
  // wait for finish
  while (!runner->Done()) {
    // sleep 0.01ms
    std::this_thread::sleep_for(std::chrono::nanoseconds(10000));
  }
  if (runner->HasException()) {
    std::rethrow_exception(runner->ExceptionPtr());
  }
  return outputs;
}

std::vector<std::pair<std::string, RTValue>> SimpleMPMCServer::process_with_tc(
    const std::unordered_map<std::string, RTValue>& feed_dict,
    uint64_t* enqueue_tc_us,
    uint64_t* real_run_tc_us) {
  std::vector<std::pair<std::string, RTValue>> outputs;
  auto runner = std::make_shared<RunnableWithTimeCost>(&feed_dict, &outputs);
  uint64_t enqueue_begin = EnvTime::Default()->NowMicros();
  auto runner_base = std::static_pointer_cast<RunnableWithTimeCost>(runner);
  // push task
  for (;;) {
    if (tasks_.try_enqueue(runner_base)) {
      break;
    }
    // queue is full, sleep 1us
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
  uint64_t enqueue_end = EnvTime::Default()->NowMicros();
  *enqueue_tc_us = enqueue_end - enqueue_begin;
  // wait for finish
  while (!runner->Done()) {
    // sleep 0.01ms
    std::this_thread::sleep_for(std::chrono::nanoseconds(10000));
  }
  *real_run_tc_us = runner->time_cost;
  if (runner->HasException()) {
    std::rethrow_exception(runner->ExceptionPtr());
  }
  return outputs;
}

}  // namespace server
}  // namespace runtime
}  // namespace matxscript
