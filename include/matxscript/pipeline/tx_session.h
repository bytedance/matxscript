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

#include <unistd.h>

#include <map>
#include <memory>
#include <mutex>

#include <matxscript/pipeline/constant_op.h>
#include <matxscript/pipeline/graph.h>
#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/pipeline/signature_constants.h>
#include <matxscript/pipeline/userdata_scoped_cache.h>
#include <matxscript/pipeline/variable_op.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/profiling_helper.h>
#include <matxscript/runtime/threadpool/i_thread_pool.h>
#include <matxscript/runtime/threadpool/thread_pool_executor.h>

namespace matxscript {
namespace runtime {

struct TXSessionOptions {
  String name = "__global__";
  bool share_scheduling_pool = false;
  bool share_compute_pool = false;
  bool enable_graph_parallel = false;
  bool enable_scheduling_pool = true;
  bool enable_device_op_parallel = false;
  bool enable_compute_pool = true;
  int32_t compute_pool_thread_nums = 8;
  int32_t scheduling_pool_thread_nums = 2;
  int32_t min_task_size_one_thread = 1;
  int32_t max_task_size_one_thread = 1;
};

struct TXSessionStepStat {
  String op;
  String op_cls;
  TimeLine time_line;
  RTValue inputs;
  RTValue output;
  RTValue attributes;
};

struct TXSessionRunMeta {
  TimeLine time_line;
  std::vector<TXSessionStepStat> step_stats;
};

extern TXSessionOptions DEFAULT_SESSION_OPTIONS;

struct TXSession {
 public:
  explicit TXSession(TXSessionOptions opt);
  TXSession() : TXSession(DEFAULT_SESSION_OPTIONS) {
  }
  virtual ~TXSession() = default;

 public:
  void Save(string_view folder, string_view name) const;
  static std::unique_ptr<TXSession> Load(string_view folder,
                                         string_view name,
                                         int device = -1,
                                         string_view version = "");

  // After fork, the child process should call this function once
  void AtForkBefore();
  void AtForkAfterInParentOrChild();

  /**
   * build a new trace and clear history
   * @param outputs
   */
  void Trace(const std::vector<const Symbol*>& outputs);
  void Trace(const Symbol* output) {
    return Trace(std::vector<const Symbol*>{output});
  }

  void SetSchedulingThreads(int32_t num = 2, bool share = false);
  void SetOpParallelismThreads(int32_t num = 2, bool share = false);
  void SetOpComputeThreads(int32_t num = 8, bool share = false);

  int64_t GetSchedulingThreads();
  int64_t GetOpParallelismThreads();
  internal::IThreadPool* GetSchedulingThreadPool();
  ThreadPoolExecutor* GetSchedulingThreadPoolExecutor();
  internal::IThreadPool* GetComputeThreadPool();
  ThreadPoolExecutor* GetComputeThreadPoolExecutor();

  /**
   * just multi-run by last trace
   * @param node
   * @param feed_dict
   * @return
   */
  std::vector<std::pair<std::string, RTValue>> Run(
      const std::unordered_map<std::string, RTValue>& feed_dict) const;

  std::vector<std::pair<std::string, RTValue>> Run(
      const std::unordered_map<std::string, RTValue>& feed_dict, TXSessionRunMeta* meta) const;

  /**
   * Each task thread will execute a session run at once
   * @param feed_dict
   * @return
   */
  std::vector<std::pair<std::string, RTValue>> Warmup(
      const std::unordered_map<std::string, RTValue>& feed_dict) const;

  /**
   * bind device by serial number
   * @param device
   */
  void SetDevice(int device);

  /**
   * get device
   * @return
   */
  int GetDevice() const;

  std::unique_ptr<Symbol> CreateVariable(std::string name, RTValue value = RTValue()) {
    auto var_op = GetVariableOp();
    MXCHECK(!name.empty()) << "variable name must be specified by user!!!";
    return VariableOp::make_symbol(var_op, std::move(name), std::move(value));
  }

  std::unique_ptr<Symbol> CreateConstant(const RTValue& val) {
    Attributes attrs;
    attrs.SetAttr("data", val);
    auto op_ptr = CreateOp("ConstantOp", attrs.ToDict());
    auto const_op_ptr = std::dynamic_pointer_cast<ConstantOp>(op_ptr);
    return ConstantOp::make_symbol(const_op_ptr);
  }

  RTValue GetAttr(const string_view& key) const;

  void SetAttr(const string_view& key, RTValue value);

  bool HasAttr(const string_view& key) const;

 public:
  std::shared_ptr<VariableOp> GetVariableOp() {
    static std::shared_ptr<VariableOp> var_op = std::make_shared<VariableOp>();
    if (!ud_cache_->Get(var_op->class_name_, var_op->name_).defined()) {
      auto ud_ref = make_userdata(var_op);
      ud_cache_->Set(var_op->class_name_, var_op->name_, std::move(ud_ref));
    }
    return var_op;
  }

  OpKernelPtr CreateOp(string_view class_name, Dict attrs, string_view cache_key = "");
  OpKernelPtr FindOp(string_view class_name, string_view cache_key);
  void Remove(string_view class_name, string_view cache_key);

  JitObjectPtr FindJitObject(string_view cache_key);

  UserDataRef FindUserData(string_view class_name, string_view cache_key);

  std::vector<std::string> InputNames() const;

  List GetOpInstanceName() const;
  void GetOpInstanceNameDfs(const OpKernelPtr& ops, List& op_instance_names) const;

  static Dict GetNestedOpAttributes(const OpKernel* op);

 private:
  class TXSessionRunnable;
  class TXSessionWarmupRunnable;
  static TXSessionStepStat MakeSessionStepStat(const NodePtr& op);
  void RunImpl(const std::unordered_map<std::string, RTValue>& feed_dict,
               std::vector<std::pair<std::string, RTValue>>& result,
               TXSessionRunMeta* meta = nullptr) const;

  void RunImplMultiThread(const std::unordered_map<std::string, RTValue>& feed_dict,
                          std::vector<std::pair<std::string, RTValue>>& result,
                          TXSessionRunMeta* meta = nullptr) const;

  void SetOutput(const std::unordered_map<std::string, RTValue>& feed_dict,
                 const ska::flat_hash_map<string_view, RTValue>& datapack,
                 std::vector<std::pair<std::string, RTValue>>& output) const;

  void BuildRunNodes();
  void BuildOutputKeys();

  void DFSCopyOp(OpKernelPtr& op);
  void TransPythonOp(OpKernelPtr& op);

  void DFSSaveOp(OpKernelPtr op,
                 string_view folder,
                 ska::flat_hash_set<const OpKernel*>& visited,
                 List& generic_ops) const;

 private:
  DLDeviceType device_type_ = kDLCPU;
  int device_ = NONE_DEVICE;
  DeviceAPI* device_api_ = nullptr;
  std::shared_ptr<Graph> graph_;
  std::vector<NodePtr> serial_nodes_;
  std::vector<std::vector<NodePtr>> parallel_nodes_;
  size_t datapack_element_size_;
  std::vector<NodeEntryPtr> outputs_;
  std::vector<std::string> output_keys_;
  std::shared_ptr<UserDataScopedCache> ud_cache_;
  std::shared_ptr<UserDataScopedCache> ud_cache_local_;
  TXSessionOptions options_;
  std::shared_ptr<internal::IThreadPool> scheduling_pool_ = nullptr;
  std::shared_ptr<ThreadPoolExecutor> scheduling_pool_executor_;
  Attributes attributes_;
  std::shared_ptr<internal::IThreadPool> compute_pool_ = nullptr;
  std::shared_ptr<ThreadPoolExecutor> compute_pool_executor_;

  friend class Graph;
};

}  // namespace runtime
}  // namespace matxscript
