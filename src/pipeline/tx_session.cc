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
#include <matxscript/pipeline/tx_session.h>

#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <matxscript/pipeline/interpreter_op.h>
#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/jit_op.h>
#include <matxscript/pipeline/node.h>
#include <matxscript/pipeline/pickle.h>
#include <matxscript/pipeline/py_torch_infer_op.h>
#include <matxscript/pipeline/python_base_op.h>
#include <matxscript/pipeline/threadpool_op.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/json_util.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/profiling_helper.h>
#include <matxscript/runtime/threadpool/lock_based_thread_pool.h>

#include <matxscript/runtime/future_wrap.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_funcs.h>

namespace matxscript {
namespace runtime {

static const char ThreadPoolOpClassName[] = "ThreadPoolOp";
static const char ComputeThreadPoolOpName[] = "ThreadPoolOp_compute_pool_0";
static const char ScheduleThreadPoolOpName[] = "ThreadPoolOp_scheduling_pool_0";

static TXSessionOptions TXSessionOptionsReadFromDict(const Dict& config) {
  TXSessionOptions sess_opts;
  sess_opts.name = config["name"].As<String>();

  // parse graph_parallel flag
  if (config.contains("enable_graph_parallel")) {
    sess_opts.enable_graph_parallel = config["enable_graph_parallel"].As<bool>();
  } else if (config.contains(U"enable_graph_parallel")) {
    sess_opts.enable_graph_parallel = config[U"enable_graph_parallel"].As<bool>();
  }
  // parse scheduling pool config
  if (config.contains("enable_scheduling_pool")) {
    sess_opts.enable_scheduling_pool = config["enable_scheduling_pool"].As<bool>();
  } else if (config.contains(U"enable_scheduling_pool")) {
    sess_opts.enable_scheduling_pool = config[U"enable_scheduling_pool"].As<bool>();
  }
  // force enable_scheduling_pool when enable graph parallel
  sess_opts.enable_scheduling_pool |= sess_opts.enable_graph_parallel;

  if (config.contains("share_scheduling_pool")) {
    sess_opts.share_scheduling_pool = config["share_scheduling_pool"].As<bool>();
  } else if (config.contains(U"share_scheduling_pool")) {
    sess_opts.share_scheduling_pool = config[U"share_scheduling_pool"].As<bool>();
  }
  if (sess_opts.enable_scheduling_pool) {
    sess_opts.enable_device_op_parallel = true;
    // pool_thread_nums is an option for old version
    if (config.contains("scheduling_pool_thread_nums")) {
      sess_opts.scheduling_pool_thread_nums = config["scheduling_pool_thread_nums"].As<int32_t>();
    } else if (config.contains(U"scheduling_pool_thread_nums")) {
      sess_opts.scheduling_pool_thread_nums = config[U"scheduling_pool_thread_nums"].As<int32_t>();
    }
    if (sess_opts.scheduling_pool_thread_nums <= 0) {
      sess_opts.scheduling_pool_thread_nums = 2;
    }
  }

  // parse compute pool config
  if (config.contains("enable_compute_pool")) {
    sess_opts.enable_compute_pool = config["enable_compute_pool"].As<bool>();
  } else if (config.contains(U"enable_compute_pool")) {
    sess_opts.enable_compute_pool = config[U"enable_compute_pool"].As<bool>();
  }
  if (config.contains("share_compute_pool")) {
    sess_opts.share_compute_pool = config["share_compute_pool"].As<bool>();
  } else if (config.contains(U"share_compute_pool")) {
    sess_opts.share_compute_pool = config[U"share_compute_pool"].As<bool>();
  }
  if (sess_opts.enable_compute_pool) {
    if (config.contains("compute_pool_thread_nums")) {
      sess_opts.compute_pool_thread_nums = config["compute_pool_thread_nums"].As<int32_t>();
    } else if (config.contains(U"compute_pool_thread_nums")) {
      sess_opts.compute_pool_thread_nums = config[U"compute_pool_thread_nums"].As<int32_t>();
    }
    if (sess_opts.compute_pool_thread_nums <= 0) {
      sess_opts.compute_pool_thread_nums = 8;
    }
  }
  return sess_opts;
}

static void TXSessionOptionsWriteToDict(const TXSessionOptions& opt, Dict& config) {
  // session name for cache
  config["name"] = opt.name;

  // multi threads
  config["share_compute_pool"] = opt.share_compute_pool;
  config["share_scheduling_pool"] = opt.share_scheduling_pool;
  config["enable_graph_parallel"] = opt.enable_graph_parallel;
  config["enable_scheduling_pool"] = opt.enable_scheduling_pool;
  config["scheduling_pool_thread_nums"] = opt.scheduling_pool_thread_nums;
  config["enable_compute_pool"] = opt.enable_compute_pool;
  config["compute_pool_thread_nums"] = opt.compute_pool_thread_nums;
}

TXSessionOptions DEFAULT_SESSION_OPTIONS;

TXSession::TXSession(TXSessionOptions opt) {
  this->options_ = std::move(opt);
  datapack_element_size_ = 0;
  graph_ = nullptr;
  ud_cache_ = std::make_shared<UserDataScopedCache>(this->options_.name);
  // thread_id + this->options_.name + new uuid
  std::random_device random_seeder;
  std::mt19937 random_rng(random_seeder());
  std::uniform_int_distribution<uint64_t> random_gen(0, UINT64_MAX);  // uniform, unbiased
  uint64_t random_val = random_gen(random_rng);
  String local_scope = "local_" + std::to_string(random_val) + "_" + this->options_.name;
  ud_cache_local_ = std::make_shared<UserDataScopedCache>(local_scope);
  // set default devices is cpu
  device_ = NONE_DEVICE;
  device_type_ = kDLCPU;
  device_api_ = DeviceAPI::Get(MATXScriptDevice{device_type_, 0});
  if (this->options_.enable_graph_parallel) {
    SetOpParallelismThreads(
        options_.scheduling_pool_thread_nums > 0 ? options_.scheduling_pool_thread_nums : 0,
        options_.share_scheduling_pool);
  } else if (this->options_.enable_scheduling_pool) {
    SetSchedulingThreads(
        options_.scheduling_pool_thread_nums > 0 ? options_.scheduling_pool_thread_nums : 0,
        options_.share_scheduling_pool);
  }
  if (this->options_.enable_compute_pool) {
    SetOpComputeThreads(
        options_.compute_pool_thread_nums > 0 ? options_.compute_pool_thread_nums : 8,
        options_.share_compute_pool);
  }
}

int64_t TXSession::GetSchedulingThreads() {
  if (scheduling_pool_) {
    return scheduling_pool_->GetThreadsNum();
  }
  return 0;
}

void TXSession::SetSchedulingThreads(int32_t num, bool share) {
  options_.share_scheduling_pool = share;
  ud_cache_->Remove(ThreadPoolOpClassName, ScheduleThreadPoolOpName);
  if (num >= 0) {
    options_.enable_scheduling_pool = true;
    if (num == 0) {
      options_.scheduling_pool_thread_nums = 2;
    } else {
      options_.scheduling_pool_thread_nums = num;
    }
    MXCHECK_GT(options_.scheduling_pool_thread_nums, 0);
    MXCHECK_LT(options_.scheduling_pool_thread_nums, 256);
    if (options_.share_scheduling_pool) {
      // use thread_pool_op
      Dict attrs;
      attrs["lock_free"] = false;
      attrs["thread_nums"] = options_.scheduling_pool_thread_nums;
      attrs["thread_name"] = Unicode(U"matx.schedule");
      auto op = this->CreateOp(ThreadPoolOpClassName, attrs, ScheduleThreadPoolOpName);
      auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
      scheduling_pool_ = pool_op->GetPool();
    } else {
      scheduling_pool_ = std::make_shared<internal::LockBasedThreadPool>(
          options_.scheduling_pool_thread_nums, "matx.schedule");
    }
    scheduling_pool_executor_ = std::make_shared<ThreadPoolExecutor>(scheduling_pool_, false);
  } else {
    options_.enable_scheduling_pool = false;
    options_.scheduling_pool_thread_nums = -1;
    scheduling_pool_ = nullptr;
    scheduling_pool_executor_ = nullptr;
  }
}

void TXSession::SetOpParallelismThreads(int32_t num, bool share) {
  this->SetSchedulingThreads(num, share);
  options_.enable_graph_parallel = options_.enable_scheduling_pool;
}

int64_t TXSession::GetOpParallelismThreads() {
  if (scheduling_pool_) {
    return scheduling_pool_->GetThreadsNum();
  }
  return 0;
}

void TXSession::SetOpComputeThreads(int32_t num, bool share) {
  ud_cache_->Remove(ThreadPoolOpClassName, ComputeThreadPoolOpName);
  options_.share_compute_pool = share;
  if (num >= 0) {
    options_.enable_compute_pool = true;
    if (num == 0) {
      options_.compute_pool_thread_nums = 2;
    } else {
      options_.compute_pool_thread_nums = num;
    }
    MXCHECK_GT(options_.compute_pool_thread_nums, 0);
    MXCHECK_LT(options_.compute_pool_thread_nums, 256);
    if (options_.share_compute_pool) {
      // use thread_pool_op
      Dict attrs;
      attrs["lock_free"] = false;
      attrs["thread_nums"] = options_.compute_pool_thread_nums;
      attrs["thread_name"] = Unicode(U"matx.compute");
      auto op = this->CreateOp(ThreadPoolOpClassName, attrs, ComputeThreadPoolOpName);
      auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
      compute_pool_ = pool_op->GetPool();
    } else {
      compute_pool_ = std::make_shared<internal::LockBasedThreadPool>(
          options_.compute_pool_thread_nums, "matx.compute");
    }
    compute_pool_executor_ = std::make_shared<ThreadPoolExecutor>(compute_pool_, false);
  } else {
    options_.enable_compute_pool = false;
    options_.compute_pool_thread_nums = -1;
    compute_pool_ = nullptr;
    compute_pool_executor_ = nullptr;
  }
}

internal::IThreadPool* TXSession::GetSchedulingThreadPool() {
  return scheduling_pool_.get();
}

ThreadPoolExecutor* TXSession::GetSchedulingThreadPoolExecutor() {
  return scheduling_pool_executor_.get();
}

internal::IThreadPool* TXSession::GetComputeThreadPool() {
  return compute_pool_.get();
}

ThreadPoolExecutor* TXSession::GetComputeThreadPoolExecutor() {
  return compute_pool_executor_.get();
}

void TXSession::Remove(string_view class_name, string_view cache_key) {
  ud_cache_->Remove(class_name, cache_key);
  ud_cache_local_->Remove(class_name, cache_key);
}

OpKernelPtr TXSession::FindOp(string_view class_name, string_view cache_key) {
  UserDataRef ud = ud_cache_local_->Get(class_name, cache_key);
  if (ud.defined()) {
    return try_get_op_kernel(ud);
  }
  ud = ud_cache_->Get(class_name, cache_key);
  return try_get_op_kernel(ud);
}

OpKernelPtr TXSession::CreateOp(string_view class_name, Dict attrs, string_view cache_key) {
  auto* reg_ptr = NativeObjectRegistry::Get(class_name);
  MXCHECK(reg_ptr != nullptr) << "Op is not registered : " << class_name;
  UserDataRef ud = ud_cache_local_->Get(class_name, cache_key);
  if (ud.defined()) {
    return check_get_op_kernel(ud);
  }
  ud = ud_cache_->Get(class_name, cache_key);
  if (ud.defined()) {
    return check_get_op_kernel(ud);
  }
  if (class_name == "VariableOp") {
    return GetVariableOp();
  }
  ud = make_op_kernel(class_name, {std::move(attrs)}, this);
  auto op_ptr = check_get_op_kernel(ud);
  // load op always same as origin
  if (!cache_key.empty()) {
    op_ptr->name_ = cache_key;
    auto* nud_ptr = dynamic_cast<NativeObject*>(ud->ud_ptr);
    nud_ptr->native_instance_name_ = op_ptr->name_;
  }
  bool share = reg_ptr->threadsafety_;
  if (class_name == "JitObject") {
#ifdef MATXSCRIPT_DISABLE_SHARE_JIT_OBJECT
    share = false;
#else
    JitObjectPtr jit_ptr = try_get_jit_object(ud);
    share &= jit_ptr->options_.share;
#endif
  }
  if (share) {
    ud_cache_->Set(class_name, op_ptr->name_, std::move(ud));
  } else {
    ud_cache_local_->Set(class_name, op_ptr->name_, std::move(ud));
  }
  return op_ptr;
}

JitObjectPtr TXSession::FindJitObject(string_view cache_key) {
  UserDataRef ud = ud_cache_local_->Get("JitObject", cache_key);
  if (ud.defined()) {
    return try_get_jit_object(ud);
  }
  ud = ud_cache_->Get("JitObject", cache_key);
  return try_get_jit_object(ud);
}

UserDataRef TXSession::FindUserData(string_view class_name, string_view cache_key) {
  auto ud = ud_cache_local_->Get(class_name, cache_key);
  if (ud.defined()) {
    return ud;
  }
  return ud_cache_->Get(class_name, cache_key);
}

namespace {
bool IsDevicesOp(const OpKernel* op) {
  return op->ClassName() == "TFInferOp" || op->ClassName() == "TorchInferOp" ||
         op->ClassName() == "TVMInferOp";
}
}  // namespace

void TXSession::SetDevice(int device) {
  device_ = device;
  device_type_ = device_ < 0 ? kDLCPU : kDLCUDA;
  device_api_ = DeviceAPI::Get(MATXScriptDevice{device_type_, device_});
  if (graph_) {
    auto& nodes = graph_->get_topo_nodes();
    for (auto& node : nodes) {
      auto op = const_cast<OpKernel*>(node->op.get());
      if (IsDevicesOp(op) && op->device_ != device) {
        op->device_ = device;
        op->Init();
      }
    }
  }
}

int TXSession::GetDevice() const {
  return device_;
}

void TXSession::TransPythonOp(OpKernelPtr& op) {
  MXCHECK(op->ClassName() == TypeNameTraits::Get<PythonBaseOp>());
  auto py_op = std::static_pointer_cast<PythonBaseOp>(op);
  MXLOG(INFO) << "[TXSession] Begin op pass: " << py_op->py_op_name << " -> "
              << py_op->pass_op_name;
  auto op_copy = CreateOp(py_op->pass_op_name, py_op->pass_op_options, py_op->name_);
  MXLOG(INFO) << "[TXSession] Finish op pass: " << py_op->py_op_name << " -> "
              << py_op->pass_op_name;
  op = op_copy;
}

void TXSession::DFSCopyOp(OpKernelPtr& op) {
  for (auto& sub_op : op->sub_ops_) {
    DFSCopyOp(sub_op);
  }
  if (op->ClassName() == TypeNameTraits::Get<PythonBaseOp>()) {
    TransPythonOp(op);
  } else {
    auto* reg_ptr = NativeObjectRegistry::Get(op->ClassName());
    MXCHECK(reg_ptr != nullptr) << "Op is not registered : " << op->ClassName();
    bool share = reg_ptr->threadsafety_;
    if (op->ClassName() == "JitObject") {
      auto obj_ptr = std::static_pointer_cast<JitObject>(op);
      share &= obj_ptr->options_.share;
    }
    if (share) {
      if (!ud_cache_->Get(op->class_name_, op->name_).defined()) {
        ud_cache_->Set(op->class_name_, op->name_, make_userdata(op));
      }
    } else {
      if (!ud_cache_local_->Get(op->class_name_, op->name_).defined()) {
        ud_cache_local_->Set(op->class_name_, op->name_, make_userdata(op));
      }
    }
  }
  if (op->ClassName() == TypeNameTraits::Get<PyTorchInferOp>()) {
    // trans impl
    MXCHECK(op->sub_ops_.size() == 1) << "internal error";
    auto& real_impl = op->sub_ops_[0];
    op->SetAttr("impl", FindUserData(real_impl->ClassName(), real_impl->name_));
    op->Init();
  }
}

void TXSession::Trace(const std::vector<const Symbol*>& outputs) {
  std::vector<NodeEntryPtr> entry_outputs;
  ska::flat_hash_set<NodePtr> visited;
  for (auto& sym : outputs) {
    auto entry = sym->GetEntry();
    NodePtr node = entry->node;
    if (visited.count(node) <= 0) {
      visited.emplace(std::move(node));
    }
    entry_outputs.push_back(entry);
    entry->exported = true;
  }
  std::vector<NodePtr> output_nodes(visited.begin(), visited.end());
  auto graph = std::make_shared<Graph>(output_nodes);

  // rebuild ops
  for (auto& node : graph->get_topo_nodes()) {
    MXLOG(INFO) << "Begin Trace Op: ClassName: " << node->op->ClassName()
                << ", Name: " << node->op->GetName();
    DFSCopyOp(node->op);
    MXLOG(INFO) << "Finish Trace Op: ClassName: " << node->op->ClassName()
                << ", Name: " << node->op->GetName();
  }

  // rebuild graph
  this->graph_ = Graph::FromGenericList(this, graph->ToGenericList());

  // rebuild output
  std::unordered_map<std::string, NodeEntryPtr> name2entry;
  for (auto& node : this->graph_->get_topo_nodes()) {
    for (auto& entry : node->outputs) {
      if (entry.source->exported) {
        name2entry.emplace(entry.source->key, entry.weak_ref.lock());
      }
    }
  }
  this->outputs_.clear();
  for (auto& raw_entry : entry_outputs) {
    MXCHECK(name2entry.find(raw_entry->key) != name2entry.end());
    this->outputs_.push_back(name2entry[raw_entry->key]);
  }
  BuildRunNodes();
}

void TXSession::BuildRunNodes() {
  // dependence analysis and group to parallel
  parallel_nodes_.clear();
  serial_nodes_ = graph_->get_topo_nodes();
  if (serial_nodes_.empty()) {
    MXLOG(FATAL) << "[TXSession:trace] compute node num is 0, do nothing!!!";
    return;
  }

  // compute datapack element size
  datapack_element_size_ = 0;
  for (auto& node : serial_nodes_) {
    datapack_element_size_ += node->outputs.size();
  }

  std::unordered_set<NodePtr> finish_nodes;
  finish_nodes.reserve(serial_nodes_.size());

  // push input to finish nodes
  for (auto& node : serial_nodes_) {
    if (node->IsVariable()) {
      finish_nodes.insert(node);
    }
  }
  for (;;) {
    bool finish = true;
    // int32_t infer_node_num = 0;
    std::vector<NodePtr> run_nodes;
    for (auto& node : serial_nodes_) {
      if (finish_nodes.find(node) == finish_nodes.end()) {
        // this node is not be processed
        finish = false;
        bool input_ready = true;
        for (auto& entry : node->inputs) {
          if (finish_nodes.find(entry->node) == finish_nodes.end()) {
            input_ready = false;
          }
        }
        if (input_ready) {
          auto op_ptr = node->op;
          /*if (IsDevicesOp(op_ptr.get())) {
            ++infer_node_num;
          }*/
          if (true /*options_.enable_parallel_infer || infer_node_num <= 1 */) {
            run_nodes.push_back(node);
          }
        }
      }
    }
    if (finish) {
      break;
    } else if (run_nodes.empty()) {
      // compute graph is bad for dependence loss
      MXTHROW << "compute graph is bad for dependence loss";
    } else {
      finish_nodes.insert(run_nodes.begin(), run_nodes.end());
      parallel_nodes_.push_back(std::move(run_nodes));
    }
  }
}

void TXSession::BuildOutputKeys() {
  bool valid = true;
  std::vector<std::string> output_keys;
  if (this->HasAttr(constants::kServingOutputSignatureDefKey)) {
    auto attr = this->GetAttr(constants::kServingOutputSignatureDefKey);
    if (attr.IsObjectRef<List>()) {
      auto output_names = attr.AsObjectRefNoCheck<List>();
      for (auto& n : output_names) {
        switch (n.type_code()) {
          case TypeIndex::kRuntimeUnicode: {
            auto ns = UnicodeHelper::Encode(n.AsNoCheck<unicode_view>());
            output_keys.emplace_back(ns.data(), ns.size());
          } break;
          case TypeIndex::kRuntimeString: {
            auto ns = n.AsNoCheck<string_view>();
            output_keys.emplace_back(ns.data(), ns.size());
          } break;
          default: {
            valid = false;
            break;
            MXLOG(INFO) << "[TXSession:BuildOutputKeys] parse output_names failed!!!";
          } break;
        }
      }
    }
  }
  if (valid) {
    output_keys_ = std::move(output_keys);
  }
}

std::vector<std::pair<std::string, RTValue>> TXSession::Run(
    const std::unordered_map<std::string, RTValue>& feed_dict) const {
  MXCHECK(graph_) << "forget trace? run must after trace!!!";
  std::vector<std::pair<std::string, RTValue>> result;
  if (options_.enable_graph_parallel && options_.enable_scheduling_pool && scheduling_pool_) {
    RunImplMultiThread(feed_dict, result);
  } else {
    RunImpl(feed_dict, result);
  }
  return result;
}

std::vector<std::pair<std::string, RTValue>> TXSession::Run(
    const std::unordered_map<std::string, RTValue>& feed_dict, TXSessionRunMeta* meta) const {
  ProfilingHelper ph(meta ? &meta->time_line : nullptr);
  MXCHECK(graph_) << "forget trace? run must after trace!!!";
  std::vector<std::pair<std::string, RTValue>> result;
  if (meta) {
    meta->step_stats.reserve(serial_nodes_.size());
  }
  if (options_.enable_graph_parallel && options_.enable_scheduling_pool && scheduling_pool_) {
    RunImplMultiThread(feed_dict, result, meta);
  } else {
    RunImpl(feed_dict, result, meta);
  }
  return result;
}

static int TXSessionRunOneNode(const NodePtr& node,
                               const std::unordered_map<std::string, RTValue>& feed_dict,
                               const ska::flat_hash_map<string_view, RTValue>& datapack,
                               ska::flat_hash_map<string_view, RTValue>* output_dict,
                               TXSessionStepStat* step_stat) {
  if (node->IsVariable()) {
    return 1;
  }
  ProfilingHelper prof_helper(step_stat ? &(step_stat->time_line) : nullptr);
  std::vector<RTView> op_feed;
  op_feed.reserve(node->inputs.size());
  for (auto& entry : node->inputs) {
    auto& node_input = entry->node;
    if (node_input->IsVariable()) {
      auto itr = feed_dict.find(entry->key);
      MXCHECK(itr != feed_dict.end()) << "[" << entry->key << "] feed value not found!!!";
      op_feed.emplace_back(itr->second);
    } else {
      auto itr = datapack.find(entry->key.view());
      MXCHECK(itr != datapack.end()) << entry->key << " not found in datapack";
      op_feed.emplace_back(itr->second);
    }
  }
  RTValue rets = node->op->Process(PyArgs(op_feed.data(), op_feed.size()));
  if (step_stat) {
    step_stat->inputs = Tuple(op_feed.data(), op_feed.data() + op_feed.size());
    step_stat->output = rets;
    if (step_stat->op.startswith("matx.pmap") || step_stat->op.startswith("matx.pstarmap") ||
        step_stat->op.startswith("matx.apply_async")) {
      if (op_feed.size() >= 1 && op_feed[0].IsObjectRef<UserDataRef>()) {
        auto parallel_callable = op_feed[0].AsNoCheck<UserDataRef>();
        auto parallel_op = try_get_op_kernel(parallel_callable);
        String func_repr;
        if (parallel_op && parallel_op->ClassName() == "JitOp") {
          func_repr = std::static_pointer_cast<JitOp>(parallel_op)->GetHumanName(false);
        } else {
          func_repr = parallel_callable.__str__().encode();
          if (func_repr.startswith("<function ")) {
            func_repr = func_repr.substr(10);
            func_repr = func_repr.substr(0, func_repr.find(" "));
          } else if (func_repr.startswith("<") && func_repr.endswith(">")) {
            auto first_space_pos = func_repr.find(" ");
            if (first_space_pos != String::npos &&
                func_repr.substr(first_space_pos + 1).startswith("object at ")) {
              func_repr = func_repr.substr(1, first_space_pos - 1);
            }
          }
        }
        auto at_pos = step_stat->op.find(" @");
        if (at_pos == String::npos) {
          step_stat->op = step_stat->op + "(" + func_repr + ", ...)";
        } else {
          step_stat->op = step_stat->op.substr(0, at_pos) + "(" + func_repr + ", ...)" +
                          step_stat->op.substr(at_pos);
        }
      }
    }
  }
  if (node->outputs.size() > 1) {
    MXCHECK(rets.IsObjectRef<Tuple>()) << "expect tuple outputs, but get: " << rets.type_name();
    Tuple adt_tuple = rets.MoveToObjectRefNoCheck<Tuple>();
    MXCHECK(adt_tuple.size() == node->outputs.size());
    for (size_t e = 0; e < node->outputs.size(); ++e) {
      output_dict->emplace(node->outputs[e].source->key.view(), std::move(adt_tuple[e]));
    }
  } else {
    MXCHECK(!node->outputs.empty());
    output_dict->emplace(node->outputs[0].source->key.view(), std::move(rets));
  }
  return 1;
}

void TXSession::SetOutput(const std::unordered_map<std::string, RTValue>& feed_dict,
                          const ska::flat_hash_map<string_view, RTValue>& datapack,
                          std::vector<std::pair<std::string, RTValue>>& output) const {
  // make output
  size_t i = 0;
  bool use_sig = output_keys_.size() == outputs_.size();
  for (auto& entry : outputs_) {
    if (entry->node->IsVariable()) {
      auto itr = feed_dict.find(entry->key);
      MXCHECK(itr != feed_dict.end()) << "[" << entry->key << "] feed value not found!!!";
      if (use_sig) {
        output.emplace_back(output_keys_[i], itr->second);
      } else {
        output.emplace_back(std::to_string(i), itr->second);
      }
    } else {
      auto itr = datapack.find(entry->key.view());
      MXCHECK(itr != datapack.end()) << "[" << entry->key << "] feed value not found!!!";
      if (use_sig) {
        output.emplace_back(output_keys_[i], itr->second);
      } else {
        output.emplace_back(std::to_string(i), itr->second);
      }
    }
    ++i;
  }
}

Dict TXSession::GetNestedOpAttributes(const OpKernel* op) {
  Dict d;
  d.reserve(op->attributes_.attrs_.size());
  for (auto& attr : op->attributes_.attrs_) {
    bool ready = false;
    if (attr.second.IsObjectRef<UserDataRef>()) {
      auto ud_attr = attr.second.AsNoCheck<UserDataRef>();
      if (ud_attr->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData) {
        auto nat_obj_ptr = static_cast<NativeObject*>(ud_attr->ud_ptr);
        if (nat_obj_ptr->is_jit_object_ || nat_obj_ptr->is_native_op_) {
          OpKernel* op_ptr = nullptr;
          if (nat_obj_ptr->opaque_ptr_) {
            op_ptr = static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get());
          } else {
            ud_attr = op->belong_to_->FindUserData(nat_obj_ptr->native_class_name_,
                                                   nat_obj_ptr->native_instance_name_);
            nat_obj_ptr = static_cast<NativeObject*>(ud_attr->ud_ptr);
            op_ptr = static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get());
          }
          MXCHECK(op_ptr != nullptr) << "internal error!!!";
          Dict op_attr;
          op_attr[U"op"] = StringHelper::Decode(op_ptr->name_);
          op_attr[U"op_cls"] = StringHelper::Decode(op_ptr->class_name_);
          op_attr[U"attributes"] = TXSession::GetNestedOpAttributes(op_ptr);
          d[attr.first.decode()] = std::move(op_attr);
          ready = true;
        }
      }
    }
    if (!ready) {
      d[attr.first.decode()] = attr.second;
    }
  }
  return d;
}

TXSessionStepStat TXSession::MakeSessionStepStat(const NodePtr& node) {
  auto* op = node->op.get();
  TXSessionStepStat stat;
  if (op->ClassName() == "JitOp") {
    stat.op = static_cast<const JitOp*>(op)->GetHumanName(true);
  } else if (op->ClassName() == "InterpreterOp") {
    stat.op = static_cast<const InterpreterOp*>(op)->GetHumanName(true);
  } else if (op->ClassName() == "VariableOp") {
    stat.op = "Input: " + node->name;
  } else if (op->ClassName() == "ConstantOp") {
    stat.op = "GetConstant";
  } else {
    stat.op = op->name_;
  }
  // stat.op = op->name_;
  stat.op_cls = op->class_name_;
  stat.attributes = TXSession::GetNestedOpAttributes(op);
  return stat;
}

void TXSession::RunImpl(const std::unordered_map<std::string, RTValue>& feed_dict,
                        std::vector<std::pair<std::string, RTValue>>& output,
                        TXSessionRunMeta* meta) const {
  if (serial_nodes_.empty()) {
    MXLOG(INFO) << "[TXSession:trace] compute node num is 0, do nothing!!!";
    return;
  }
  TXSessionStepStat* step_stats = nullptr;
  if (meta) {
    for (size_t i = 0; i < serial_nodes_.size(); ++i) {
      meta->step_stats.emplace_back(TXSession::MakeSessionStepStat(serial_nodes_[i]));
    }
    step_stats = meta->step_stats.data();
  }

  ska::flat_hash_map<string_view, RTValue> datapack;
  datapack.reserve(datapack_element_size_);
  for (auto& node : serial_nodes_) {
    TXSessionRunOneNode(node, feed_dict, datapack, &datapack, step_stats);
    if (step_stats) {
      ++step_stats;
    }
  }

  // make output
  SetOutput(feed_dict, datapack, output);
}

std::vector<std::pair<std::string, RTValue>> TXSession::Warmup(
    const std::unordered_map<std::string, RTValue>& feed_dict) const {
  MXCHECK(graph_) << "forget trace? warmup must after trace!!!";
  if (options_.enable_graph_parallel && options_.enable_scheduling_pool && scheduling_pool_) {
    // MXLOG(INFO) << "[TXSession:Warmup] begin warmup in parallel mode...";
    // warmup in every thread at once
    int32_t num_threads = scheduling_pool_->GetThreadsNum();
    std::mutex control_mutex;
    int32_t num_finish = 0;
    std::vector<internal::IRunnablePtr> tasks;
    tasks.reserve(num_threads);
    for (int32_t i = 0; i < num_threads; ++i) {
      auto warm_runnable = std::make_shared<TXSessionWarmupRunnable>(
          &control_mutex, &num_finish, num_threads, this, feed_dict);
      tasks.emplace_back(std::dynamic_pointer_cast<internal::IRunnable>(std::move(warm_runnable)));
    }
    scheduling_pool_->EnqueueBulk(tasks);
    // warmup in main thread
    auto result = Run(feed_dict);
    // wait all tasks finish
    internal::IThreadPool::WaitBulk(tasks);
    // MXLOG(INFO) << "[TXSession:Warmup] finish warmup in parallel mode...";
    return result;
  } else {
    // MXLOG(INFO) << "[TXSession:Warmup] warmup in serial mode...";
    return Run(feed_dict);
  }
}

std::vector<std::string> TXSession::InputNames() const {
  std::vector<std::string> result;
  if (!HasAttr(String("input_names"))) {
    return result;
  }
  auto ll = GetAttr(String("input_names")).AsObjectRef<List>();
  result.reserve(ll.size());
  for (const auto& item : ll) {
    result.push_back(item.As<Unicode>().encode().operator std::string());
  }
  return result;
}

void TXSession::GetOpInstanceNameDfs(const OpKernelPtr& op, List& op_instance_names) const {
  for (const auto& sub_op : op->sub_ops_) {
    GetOpInstanceNameDfs(sub_op, op_instance_names);
  }
  List jit_op_names = op_instance_names[0].As<List>();
  List native_op_names = op_instance_names[1].As<List>();

  if (op->ClassName() == TypeNameTraits::Get<JitOp>() ||
      op->ClassName() == TypeNameTraits::Get<JitObject>()) {
    jit_op_names.append(op->GetName().decode());
  } else {
    static std::unordered_set<String> builtin_op_class_names{
        "VariableOp",
        "ConstantOp",
    };
    if (!builtin_op_class_names.count(op->ClassName())) {
      native_op_names.append(op->GetName().decode());
    }
  }
}

List TXSession::GetOpInstanceName() const {
  List ret;
  ret.reserve(2);
  ret.append(List());
  ret.append(List());
  for (const auto& node : serial_nodes_) {
    GetOpInstanceNameDfs(node->op, ret);
  }

  return ret;
}

class TXSession::TXSessionWarmupRunnable : public internal::LockBasedRunnable {
 public:
  TXSessionWarmupRunnable(std::mutex* wait_finish_lock,
                          int32_t* p_num_finish,
                          int32_t worker_num,
                          const TXSession* sess,
                          const std::unordered_map<std::string, RTValue>& feed_dict)
      : wait_finish_lock_(wait_finish_lock),
        p_num_finish_(p_num_finish),
        worker_num_(worker_num),
        sess_(sess),
        feed_dict_(feed_dict) {
    device_.device_type = sess->device_type_;
    device_.device_id = sess_->device_;
    auto* dev_api = sess_->device_api_;
    stream_ = dev_api->GetSharedCurrentThreadStream(device_);
  }

  void RunImpl() override {
    // follow main thread stream
    auto* dev_api = sess_->device_api_;
    auto stream = dev_api->GetSharedCurrentThreadStream(device_);
    if (stream.get() != stream_.get()) {
      dev_api->SetCurrentThreadStream(device_, stream_);
    }

    auto tid = std::this_thread::get_id();
    std::this_thread::sleep_for(
        std::chrono::microseconds(std::hash<std::thread::id>{}(tid) % 1000));
    // MXLOG(INFO) << "[TXSession:Warmup] warmup in thread: " << tid;
    std::vector<std::pair<std::string, RTValue>> result;
    sess_->RunImpl(feed_dict_, result);
    int32_t num_finish = 0;
    {
      std::lock_guard<std::mutex> lock(*wait_finish_lock_);
      ++(*p_num_finish_);
      num_finish = *p_num_finish_;
    }
    while (num_finish != worker_num_) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      // maybe no need lock
      std::lock_guard<std::mutex> lock(*wait_finish_lock_);
      num_finish = *p_num_finish_;
    }
  }

 private:
  const std::unordered_map<std::string, RTValue> feed_dict_;
  std::mutex* wait_finish_lock_;
  int32_t* p_num_finish_;
  int32_t worker_num_;
  const TXSession* sess_;
  MATXScriptDevice device_;
  std::shared_ptr<void> stream_;
};

class TXSession::TXSessionRunnable : public internal::LockBasedRunnable {
 public:
  TXSessionRunnable(const NodePtr* node,
                    int32_t node_num,
                    const std::unordered_map<std::string, RTValue>& feed_dict,
                    const ska::flat_hash_map<string_view, RTValue>& datapack,
                    ska::flat_hash_map<string_view, RTValue>* output_dict,
                    TXSessionStepStat* step_stat,
                    const TXSession* sess)
      : p_node_(node),
        node_num_(node_num),
        p_feed_dict_(&feed_dict),
        p_datapack_(&datapack),
        p_output_dict_(output_dict),
        step_stat_(step_stat) {
    sess_ = sess;
    device_.device_type = sess_->device_type_;
    device_.device_id = sess_->device_;
    stream_ = sess_->device_api_->GetSharedCurrentThreadStream(device_);

    if (node_num < 1) {
      MXCHECK_GT(node_num, 1);
    }
  }

  void RunImpl() override {
    // follow main thread stream
    auto stream = sess_->device_api_->GetSharedCurrentThreadStream(device_);
    if (stream.get() != stream_.get()) {
      sess_->device_api_->SetCurrentThreadStream(device_, stream_);
    }

    if (node_num_ == 1) {
      p_output_dict_->reserve(p_node_[0]->outputs.size());
      TXSessionRunOneNode(p_node_[0], *p_feed_dict_, *p_datapack_, p_output_dict_, step_stat_);
    } else {
      int ret_total = 0;
      int ele_num = 0;
      for (int32_t i = 0; i < node_num_; ++i) {
        ele_num += p_node_[i]->outputs.size();
      }
      p_output_dict_->reserve(ele_num);
      for (int32_t i = 0; i < node_num_; ++i) {
        ret_total += TXSessionRunOneNode(
            p_node_[i], *p_feed_dict_, *p_datapack_, p_output_dict_, step_stat_ + i);
      }
      return;
    }
  }

 private:
  const NodePtr* p_node_;
  int32_t node_num_;
  const std::unordered_map<std::string, RTValue>* p_feed_dict_;
  const ska::flat_hash_map<string_view, RTValue>* p_datapack_;
  ska::flat_hash_map<string_view, RTValue>* p_output_dict_;
  TXSessionStepStat* step_stat_ = nullptr;
  MATXScriptDevice device_;
  std::shared_ptr<void> stream_;
  const TXSession* sess_;
};

void TXSession::RunImplMultiThread(const std::unordered_map<std::string, RTValue>& feed_dict,
                                   std::vector<std::pair<std::string, RTValue>>& output,
                                   TXSessionRunMeta* meta) const {
  if (parallel_nodes_.empty()) {
    MXLOG(INFO) << "[TXSession:trace] compute node num is 0, do nothing!!!";
    return;
  }
  int32_t parallel_num = scheduling_pool_->GetThreadsNum();

  ska::flat_hash_map<string_view, RTValue> datapack;
  datapack.reserve(datapack_element_size_);
  for (auto& run_nodes : parallel_nodes_) {
    if (run_nodes.empty()) {
      continue;
    }
    if (run_nodes.size() == 1) {
      TXSessionStepStat* step_stats = nullptr;
      if (meta) {
        size_t last_idx = meta->step_stats.size();
        meta->step_stats.emplace_back(TXSession::MakeSessionStepStat(run_nodes[0]));
        step_stats = meta->step_stats.data() + last_idx;
      }
      TXSessionRunOneNode(run_nodes[0], feed_dict, datapack, &datapack, step_stats);
    } else {
      TXSessionStepStat* step_stats = nullptr;
      if (meta) {
        size_t last_idx = meta->step_stats.size();
        for (size_t i = 0; i < run_nodes.size(); ++i) {
          meta->step_stats.emplace_back(TXSession::MakeSessionStepStat(run_nodes[i]));
        }
        step_stats = meta->step_stats.data() + last_idx;
      }
      std::vector<ska::flat_hash_map<string_view, RTValue>> output_dicts;
      std::vector<internal::IRunnablePtr> runnables;
      int run_nodes_num = run_nodes.size();
      int step = (run_nodes_num + parallel_num - 1) / parallel_num;
      if (step < options_.min_task_size_one_thread) {
        step = options_.min_task_size_one_thread;
      }
      if (step > options_.max_task_size_one_thread) {
        step = options_.max_task_size_one_thread;
      }
      int runables_num = (run_nodes_num + step - 1) / step;
      runnables.reserve(runables_num);
      output_dicts.resize(runables_num, ska::flat_hash_map<string_view, RTValue>());
      for (int i = 0; i < runables_num; ++i) {
        auto real_step = step;
        if (i + 1 == runables_num) {
          real_step = run_nodes_num - i * step;
        }
        auto runner = std::make_shared<TXSessionRunnable>(run_nodes.data() + step * i,
                                                          real_step,
                                                          feed_dict,
                                                          datapack,
                                                          &output_dicts[i],
                                                          step_stats,
                                                          this);
        runnables.push_back(std::dynamic_pointer_cast<internal::IRunnable>(runner));
        if (step_stats) {
          step_stats += real_step;
        }
      }
      auto last_runnable = runnables.back();
      runnables.pop_back();
      if (!runnables.empty()) {
        scheduling_pool_->EnqueueBulk(runnables);
      }
      last_runnable->Run();
      std::exception_ptr eptr;
      for (auto& task : runnables) {
        try {
          task->Wait();
        } catch (...) {
          if (!eptr) {
            eptr = std::current_exception();
          }
        }
      }
      try {
        last_runnable->Wait();
      } catch (...) {
        if (!eptr) {
          eptr = std::current_exception();
        }
      }
      if (eptr) {
        std::rethrow_exception(eptr);
      }
      for (auto& output_dict : output_dicts) {
        datapack.insert(output_dict.begin(), output_dict.end());
      }
    }
  }
  // make output
  SetOutput(feed_dict, datapack, output);
}

void TXSession::DFSSaveOp(OpKernelPtr op,
                          string_view folder,
                          ska::flat_hash_set<const OpKernel*>& visited,
                          List& generic_ops) const {
  for (auto& sub_op : op->sub_ops_) {
    DFSSaveOp(sub_op, folder, visited, generic_ops);
  }
  List sub_op_names;
  for (auto& sub_op : op->sub_ops_) {
    Dict generic_sub_op;
    generic_sub_op["op"] = String(sub_op->ClassName());
    generic_sub_op["name"] = String(sub_op->GetName());
    sub_op_names.push_back(generic_sub_op);
  }
  if (visited.count(op.get()) <= 0) {
    MXLOG(INFO) << "[TXSession] begin save op: " << op->GetName();
    visited.emplace(op.get());
    Dict generic_op;
    generic_op["op"] = String(op->ClassName());
    generic_op["name"] = String(op->GetName());
    // generic_op["sub_ops"] = std::move(sub_op_names);
    op->Bundle(folder);
    generic_op["attrs"] = op->attributes_.ToDict();
    generic_ops.push_back(std::move(generic_op));
    MXLOG(INFO) << "[TXSession] finish save op : " << op->GetName();
  }
}

void TXSession::Save(string_view folder, string_view name) const {
  FileUtil::Mkdir(folder);

  Dict generic_session;

  // serialization session options
  TXSessionOptionsWriteToDict(options_, generic_session);

  // serialization ops
  List generic_ops;
  MXLOG(INFO) << "[TXSession] begin save ops...";
  ska::flat_hash_set<const OpKernel*> visited;
  for (auto& node : graph_->get_topo_nodes()) {
    DFSSaveOp(node->op, folder, visited, generic_ops);
  }
  generic_session["ops"] = std::move(generic_ops);
  MXLOG(INFO) << "[TXSession] finish save ops";

  // serialization graph
  MXLOG(INFO) << "[TXSession] begin save graph...";
  List generic_graph = graph_->ToGenericList();
  generic_session["graph"] = std::move(generic_graph);
  MXLOG(INFO) << "[TXSession] finish save graph...";

  // serialization outputs
  MXLOG(INFO) << "[TXSession] begin save outputs...";
  List generic_outputs;
  for (auto& entry : outputs_) {
    generic_outputs.push_back(entry->key);
  }
  generic_session["outputs"] = std::move(generic_outputs);
  MXLOG(INFO) << "[TXSession] finish save outputs...";

  // session attribute
  generic_session["g_attr"] = attributes_.ToDict();

  // pickle session
  rapidjson::Document sess_config = pickle::ToJsonStruct(RTView(generic_session));
  auto ss_conf = JsonUtil::ToString(&sess_config, true);
  std::string config_path;
  if (folder.empty()) {
    config_path = "./" + std::string(name.data(), name.size());
  } else {
    config_path =
        std::string(folder.data(), folder.size()) + "/" + std::string(name.data(), name.size());
  }
  std::ofstream fc(config_path);
  MXCHECK(!fc.fail()) << "open " << config_path << " failed!";
  fc << ss_conf;
  fc.close();
}

std::unique_ptr<TXSession> TXSession::Load(string_view folder,
                                           string_view name,
                                           int device,
                                           string_view version) {
  String folder_fix;
  String config_path;
  if (folder.empty()) {
    folder_fix = "./";
  } else {
    if (folder.back() == '/') {
      folder_fix = folder;
    } else {
      folder_fix = String(folder) + "/";
    }
  }
  config_path = folder_fix + String(name);
  rapidjson::Document config;
  MXCHECK(JsonUtil::FromFile(config_path, config));
  Dict generic_session = pickle::FromJsonStruct(config).As<Dict>();

  TXSessionOptions sess_opts = TXSessionOptionsReadFromDict(generic_session);
  sess_opts.name = sess_opts.name + "_" + version;
  std::unique_ptr<TXSession> sess(new TXSession(std::move(sess_opts)));

  sess->SetDevice(device);
  // init ops
  MXCHECK(generic_session.contains("ops")) << "ops not found in config!";
  MXCHECK(generic_session["ops"].IsObjectRef<List>()) << "ops is not array type";
  for (const auto& generic_op : generic_session["ops"].AsObjectRef<List>()) {
    MXCHECK(generic_op.IsObjectRef<Dict>());
    Dict op_obj = generic_op.AsObjectRef<Dict>();
    // List sub_ops = op_obj.get_item("sub_ops");
    String class_name = op_obj.get_item("op").As<String>();
    String op_name = op_obj.get_item("name").As<String>();
    try {
      Dict op_attrs = op_obj.get_item("attrs").As<Dict>();
      op_attrs[String(PREFIX_KEY)] = String(folder_fix);
      auto op = sess->CreateOp(class_name, op_attrs, op_name);
    } catch (const std::exception& ex) {
      MXCHECK(false) << "Initialize op " << class_name << ", name: " << op_name
                     << " failed. with exception:\n"
                     << ex.what();
    }
    MXLOG(INFO) << "build and Initialize op " << op_name;
  }

  // init graph
  MXCHECK(generic_session.contains("graph")) << "graph not exist in json config!";
  MXCHECK(generic_session["graph"].IsObjectRef<List>()) << "graph is not array type";
  sess->graph_ = Graph::FromGenericList(sess.get(), generic_session["graph"].As<List>());
  MXLOG(INFO) << "[INIT] init " << sess->graph_->get_topo_nodes().size() << " compute nodes";

  // build output
  MXCHECK(generic_session.contains("outputs")) << "outputs not exist in json config!";
  MXCHECK(generic_session["outputs"].IsObjectRef<List>()) << "outputs is not array type";
  auto nodes = sess->graph_->get_output_nodes();
  ska::flat_hash_map<string_view, NodeEntryPtr> name2entry;
  for (auto& node : nodes) {
    for (auto& entry : node->outputs) {
      if (entry.source->exported) {
        name2entry.emplace(entry.source->key.view(), entry.weak_ref.lock());
      }
    }
  }
  for (const auto& output : generic_session["outputs"].AsObjectRef<List>()) {
    MXCHECK(output.IsString());
    auto out_key = output.As<String>();
    MXCHECK(name2entry.find(out_key.view()) != name2entry.end());
    sess->outputs_.push_back(name2entry[out_key.view()]);
  }
  if (sess->outputs_.empty()) {
    MXLOG(INFO) << "output symbol is empty, reset to last node's output symbols";
    // set last node as output
    for (auto& entry : nodes.back()->outputs) {
      sess->outputs_.push_back(entry.weak_ref.lock());
    }
  }
  if (generic_session.contains("g_attr")) {
    auto generic_attrs = generic_session["g_attr"].AsObjectRef<Dict>();
    sess->attributes_ = Attributes::FromDict(generic_attrs);
  }
  sess->BuildRunNodes();
  sess->BuildOutputKeys();
  return std::move(sess);
}

void TXSession::AtForkBefore() {
  // After fork, the child process inherits the data-structures of the parent
  // process' thread-pool, but since those threads don't exist, the thread-pool
  // will be corrupt. So we close these threads before fork.
  if (scheduling_pool_) {
    if (options_.share_scheduling_pool) {
      auto op = this->FindOp(ThreadPoolOpClassName, ScheduleThreadPoolOpName);
      if (op) {
        auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
        pool_op->AtForkBefore();
      }
    }
    scheduling_pool_ = nullptr;
    scheduling_pool_executor_ = nullptr;
  }
  if (compute_pool_) {
    if (options_.share_compute_pool) {
      auto op = this->FindOp(ThreadPoolOpClassName, ComputeThreadPoolOpName);
      if (op) {
        auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
        pool_op->AtForkBefore();
      }
    }
    compute_pool_ = nullptr;
    compute_pool_executor_ = nullptr;
  }
}

void TXSession::AtForkAfterInParentOrChild() {
  // After fork, the child process inherits the data-structures of the parent
  // process' thread-pool, but since those threads don't exist, the thread-pool
  // is corrupt. So reinitialize the thread pool here in order to prevent segfaults.
  if (options_.scheduling_pool_thread_nums > 0) {
    if (options_.share_scheduling_pool) {
      auto op = this->FindOp(ThreadPoolOpClassName, ScheduleThreadPoolOpName);
      if (op) {
        auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
        pool_op->AtForkAfterInParentOrChild();
        scheduling_pool_ = pool_op->GetPool();
      }
    } else {
      scheduling_pool_ = std::make_shared<internal::LockBasedThreadPool>(
          options_.scheduling_pool_thread_nums, "matx.schedule");
    }
    scheduling_pool_executor_ = std::make_shared<ThreadPoolExecutor>(scheduling_pool_, false);
  }
  if (options_.compute_pool_thread_nums > 0) {
    if (options_.share_compute_pool) {
      auto op = this->FindOp(ThreadPoolOpClassName, ComputeThreadPoolOpName);
      if (op) {
        auto pool_op = std::dynamic_pointer_cast<ThreadPoolOp>(op);
        pool_op->AtForkAfterInParentOrChild();
        compute_pool_ = pool_op->GetPool();
      }
    } else {
      compute_pool_ = std::make_shared<internal::LockBasedThreadPool>(
          options_.compute_pool_thread_nums, "matx.compute");
    }
    compute_pool_executor_ = std::make_shared<ThreadPoolExecutor>(compute_pool_, false);
  }
}

void TXSession::SetAttr(const string_view& key, RTValue value) {
  attributes_.SetAttr(key, std::move(value));
}

RTValue TXSession::GetAttr(const string_view& key) const {
  return attributes_.GetAttr<RTValue>(key);
}

bool TXSession::HasAttr(const string_view& key) const {
  // for stable abi, do not change attributes_.HasAttr
  return const_cast<TXSession*>(this)->attributes_.HasAttr(key);
}

RTView GetClosureVar(void* session_handle, const string_view& cls, const string_view& name) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  return RTView(sess->FindUserData(cls, name));
}

List ParallelMap(const UserDataRef& func, const List& inputs, void* session_handle) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  auto* executor = sess ? sess->GetComputeThreadPoolExecutor() : nullptr;
  if (executor) {
    return executor->ParallelFor(func, inputs);
  } else {
    List result;
    for (auto& d : inputs) {
      result.push_back(func.generic_call(PyArgs(&d, 1)));
    }
    return result;
  }
}

Tuple ParallelMap(const UserDataRef& func, const Tuple& inputs, void* session_handle) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  auto* executor = sess ? sess->GetComputeThreadPoolExecutor() : nullptr;
  if (executor) {
    return executor->ParallelFor(func, inputs);
  } else {
    auto output_node = make_inplace_array_object<TupleNode, TupleNode::value_type>(inputs.size());
    output_node->size = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      output_node->EmplaceInit(i, func.generic_call(PyArgs(inputs.begin() + i, 1)));
      // Only increment size after the initialization succeeds
      output_node->size++;
    }
    return Tuple(std::move(output_node));
  }
}

RTValue ParallelMap(const UserDataRef& func, const Any& inputs, void* session_handle) {
  switch (inputs.type_code()) {
    case TypeIndex::kRuntimeTuple: {
      return ParallelMap(func, inputs.AsNoCheck<Tuple>(), session_handle);
    } break;
    case TypeIndex::kRuntimeList: {
      return ParallelMap(func, inputs.AsNoCheck<List>(), session_handle);
    } break;
    default: {
      THROW_PY_TypeError("matx.pmap: expect the second argument is list or tuple, but get'",
                         inputs.type_name(),
                         "'");
      return None;
    } break;
  }
}

template <typename InputArgType>
static RTValue ParallelStarMap_UnpackCall(const UserDataRef& func, const InputArgType& d) {
  switch (d.type_code()) {
    case TypeIndex::kRuntimeList: {
      auto args = d.template AsObjectRefNoCheck<List>();
      return func.generic_call(PyArgs(args.data(), args.size()));
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto args = d.template AsObjectRefNoCheck<Tuple>();
      return func.generic_call(PyArgs(args.begin(), args.size()));
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto num_args = kernel_object___len__(d);
      Iterator iterable = Kernel_Iterable::make(d);
      std::vector<RTValue> args;
      args.reserve(num_args);
      bool has_next = iterable.HasNext();
      while (has_next) {
        args.emplace_back(iterable.Next(&has_next));
      }
      return func.generic_call(PyArgs(args.data(), args.size()));
    } break;
    default: {
      MXTHROW << "matx.pstarmap(f, iterable) expect iterable[i] is list or tuple, but get "
              << d.type_name();
    } break;
  }
  return None;
}

List ParallelStarMap(const UserDataRef& func, const List& inputs, void* session_handle) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  auto* executor = sess ? sess->GetComputeThreadPoolExecutor() : nullptr;
  if (executor) {
    return executor->ParallelStarMap(func, inputs);
  } else {
    List result;
    for (auto& d : inputs) {
      result.push_back(ParallelStarMap_UnpackCall(func, d));
    }
    return result;
  }
}

Tuple ParallelStarMap(const UserDataRef& func, const Tuple& inputs, void* session_handle) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  auto* executor = sess ? sess->GetComputeThreadPoolExecutor() : nullptr;
  if (executor) {
    return executor->ParallelStarMap(func, inputs);
  } else {
    auto output_node = make_inplace_array_object<TupleNode, TupleNode::value_type>(inputs.size());
    output_node->size = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      output_node->EmplaceInit(i, ParallelStarMap_UnpackCall(func, inputs[i]));
      // Only increment size after the initialization succeeds
      output_node->size++;
    }
    return Tuple(std::move(output_node));
  }
}

RTValue ParallelStarMap(const UserDataRef& func, const Any& inputs, void* session_handle) {
  switch (inputs.type_code()) {
    case TypeIndex::kRuntimeTuple: {
      return ParallelStarMap(func, inputs.AsNoCheck<Tuple>(), session_handle);
    } break;
    case TypeIndex::kRuntimeList: {
      return ParallelStarMap(func, inputs.AsNoCheck<List>(), session_handle);
    } break;
    default: {
      THROW_PY_TypeError("matx.pstarmap: expect the second argument is list or tuple, but get'",
                         inputs.type_name(),
                         "'");
      return None;
    } break;
  }
}

RTValue ApplyAsync(const UserDataRef& func, const PyArgs& inputs, void* session_handle) {
  auto* sess = reinterpret_cast<TXSession*>(session_handle);
  auto* executor = sess ? sess->GetSchedulingThreadPoolExecutor() : nullptr;
  if (executor) {
    return executor->ApplyAsync(func, inputs);
  } else {
    auto result = func->generic_call(inputs);
    return Future::make_future_udref([r = std::move(result)]() mutable { return r; });
  }
}

}  // namespace runtime
}  // namespace matxscript
