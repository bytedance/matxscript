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

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/threadpool/mpmc_bounded_queue.h>

namespace matxscript {
namespace runtime {
namespace server {

class SimpleMPMCServer {
  class Runnable {
   public:
    Runnable(const std::unordered_map<std::string, RTValue>* inputs,
             std::vector<std::pair<std::string, RTValue>>* outputs)
        : inputs(inputs), outputs(outputs){};
    virtual void Run(const TXSession* sess_ptr) {
      try {
        *outputs = sess_ptr->Run(*inputs);
      } catch (...) {
        throw_exception_ = true;
        except_ptr_ = std::current_exception();
      }
    }
    void SetDone() {
      finish_ = true;
    }
    bool Done() const {
      return finish_;
    }

    bool HasException() const {
      return throw_exception_;
    }

    std::exception_ptr ExceptionPtr() const {
      return except_ptr_;
    }

   private:
    volatile bool finish_ = false;
    const std::unordered_map<std::string, RTValue>* inputs;
    std::vector<std::pair<std::string, RTValue>>* outputs;

    bool throw_exception_ = false;
    std::exception_ptr except_ptr_ = nullptr;
  };
  using RunnablePtr = std::shared_ptr<Runnable>;

  struct RunnableWithTimeCost : public Runnable {
    RunnableWithTimeCost(const std::unordered_map<std::string, RTValue>* inputs,
                         std::vector<std::pair<std::string, RTValue>>* outputs)
        : Runnable(inputs, outputs) {
    }
    uint64_t time_cost = 0;

    void Run(const TXSession* sess_ptr) override {
      auto begin = EnvTime::Default()->NowMicros();
      Runnable::Run(sess_ptr);
      auto end = EnvTime::Default()->NowMicros();
      time_cost = end - begin;
    }
  };

 public:
  explicit SimpleMPMCServer(std::vector<std::shared_ptr<TXSession>> handlers,
                            std::string name_prefix);
  ~SimpleMPMCServer() {
    stop();
  }

  void start();
  void stop();

  std::vector<std::pair<std::string, RTValue>> process(
      const std::unordered_map<std::string, RTValue>& feed_dict);

  std::vector<std::pair<std::string, RTValue>> process_with_tc(
      const std::unordered_map<std::string, RTValue>& feed_dict,
      uint64_t* enqueue_tc_us,
      uint64_t* real_run_tc_us);

 protected:
  static void ThreadEntry(SimpleMPMCServer* pool, TXSession* sess_ptr, const std::string& name);

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers_;
  std::vector<std::shared_ptr<TXSession>> handlers_;
  // the task queue
  ::matxscript::runtime::MPMCBoundedQueue<RunnablePtr> tasks_;
  // stop flag
  bool stop_ = false;
  std::string name_;
};

}  // namespace server
}  // namespace runtime
}  // namespace matxscript
