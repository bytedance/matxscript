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
#include <memory>
#include <vector>

#include <matxscript/pipeline/global_unique_index.h>
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/pipeline/python_base_op.h>
#include <matxscript/pipeline/symbolic_executor.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/*********************************************************************
 * trace state
 *********************************************************************/
static bool __TRACE_STATE__ = false;
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SetTraceState").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SetTraceState] Expect 1 arguments but get " << args.size();
  bool state = args[0].As<bool>();
  __TRACE_STATE__ = state;
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.GetTraceState").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[GetTraceState] Expect 0 arguments but get " << args.size();
  return __TRACE_STATE__;
});

static bool __OP_INIT_STATE__ = false;
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SetOpInitState").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SetOpInitState] Expect 1 arguments but get " << args.size();
  bool state = args[0].As<bool>();
  __OP_INIT_STATE__ = state;
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.GetOpInitState").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[GetOpInitState] Expect 0 arguments but get " << args.size();
  return __OP_INIT_STATE__;
});

/*********************************************************************
 * log
 *********************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SetLoggerLevel").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SetLoggerLevel] Expect 1 arguments but get " << args.size();
  int64_t level = args[0].As<int64_t>();
  return None;
});

/*********************************************************************
 * Operator
 *********************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("pipeline.ListAllOpNames").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[ListAllOpNames] Expect 0 arguments but get " << args.size();
  auto names = NativeObjectRegistry::ListNames();
  List result;
  for (auto& name : names) {
    if (NativeObjectRegistry::Get(name)->is_native_op_) {
      result.append(String(name).decode());
    }
  }
  return result;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.CreateNativeOp").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[CreateNativeOp] Expect 3 arguments but get " << args.size();
  void* sess = args[0].As<void*>();
  String op_cls = args[1].As<String>();
  Dict config = args[2].As<Dict>();
  auto sess_ptr = static_cast<TXSession*>(sess);
  auto op_ptr = sess_ptr->CreateOp(op_cls, config);
  return sess_ptr->FindUserData(op_ptr->ClassName(), op_ptr->GetName());
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.GetNativeOpHandle").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[GetNativeOpHandle] Expect 1 arguments but get " << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  void* handle = check_get_op_kernel(ud_ref).get();
  return handle;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.FreeNativeOp").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[FreeNativeOp] Expect 2 arguments but get " << args.size();
  void* sess = args[0].As<void*>();
  UserDataRef ud_ref = args[1].As<UserDataRef>();
  auto op_ptr = check_get_op_kernel(ud_ref);
  static_cast<TXSession*>(sess)->Remove(op_ptr->ClassName(), op_ptr->GetName());
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.OpHandleGetName").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[OpHandleGetName] Expect 1 arguments but get " << args.size();
  UserDataRef ud = args[0].As<UserDataRef>();
  OpKernelPtr op_ptr = check_get_op_kernel(ud);
  return String(op_ptr->GetName()).decode();
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.OpKernelProcess").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1) << "[OpKernelProcess] Expect 1 or more arguments but get "
                             << args.size();
  UserDataRef ud = args[0].As<UserDataRef>();
  OpKernelPtr op_ptr = check_get_op_kernel(ud);
  return op_ptr->Process(PyArgs(args.begin(), args.size() - 1));
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolicExecutor_Compose")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK_GE(args.size(), 2) << "[SymbolicExecutor_Compose] Expect 2 or more arguments but get "
                                 << args.size();
      UserDataRef ud = args[0].As<UserDataRef>();
      int64_t output_num = args[1].As<int64_t>();
      OpKernelPtr op_ptr = check_get_op_kernel(ud);
      int num_args = args.size() - 2;
      std::vector<const Symbol*> args_sym;
      args_sym.reserve(num_args);
      for (uint32_t i = 2; i < args.size(); ++i) {
        Symbol* sym = static_cast<Symbol*>(args[i].As<void*>());
        args_sym.push_back(sym);
      }
      auto out_syms = SymbolicExecutor::Compose(std::move(op_ptr), args_sym, output_num);
      std::vector<RTValue> out_sym_ptrs;
      out_sym_ptrs.reserve(out_syms.size());
      for (auto& sym : out_syms) {
        out_sym_ptrs.push_back((void*)sym.release());
      }
      return Tuple(std::make_move_iterator(out_sym_ptrs.begin()),
                   std::make_move_iterator(out_sym_ptrs.end()));
    });

/*********************************************************************
 * variable and constant
 *********************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("pipeline.CreateVariable").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[CreateVariable] Expect 3 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  Unicode name = args[1].As<Unicode>();
  RTValue data = args[2].As<RTValue>();
  auto sess = static_cast<TXSession*>(handle);
  return sess->CreateVariable(name.encode(), data).release();
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.CreateConstant").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[CreateConstant] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  RTValue data = args[1].As<RTValue>();
  std::string name = GlobalUniqueIndex::instance()->gen_uniq_name("Constant", data.type_name());
  auto sess = static_cast<TXSession*>(handle);
  Attributes attr;
  attr.SetAttr("data", std::move(data));
  auto const_op_ptr = std::make_shared<ConstantOp>();
  const_op_ptr->SetBelongTo(sess);
  const_op_ptr->Initialize(std::move(attr));
  auto s_ptr = ConstantOp::make_symbol(const_op_ptr);
  return s_ptr.release();
});

/*********************************************************************
 * Symbol
 *********************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolFree").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SymbolFree] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  delete static_cast<Symbol*>(handle);
  return None;
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolGetName").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SymbolGetName] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto* sym = static_cast<Symbol*>(handle);
  return sym->GetEntry()->Name().decode();
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolGetKey").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SymbolGetKey] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto* sym = static_cast<Symbol*>(handle);
  return sym->GetEntry()->key.decode();
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolGetVal").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SymbolGetVal] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto* sym = static_cast<Symbol*>(handle);
  return sym->GetEntry()->data;
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.SymbolSetVal").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[SymbolSetFirstOutVal] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  RTValue data = args[1].As<RTValue>();
  auto sym = static_cast<Symbol*>(handle);
  sym->GetEntry()->data = std::move(data);
  return None;
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.GetOpInstanceName").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[GetOpInstanceName] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  return sess->GetOpInstanceName();
});

/*********************************************************************
 * TXSession
 *********************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("pipeline.CreateTXSessionHandle").set_body([](PyArgs args) -> RTValue {
  MXCHECK_LE(args.size(), 1) << "[CreateTXSessionHandle] Expect 0 or 1 arguments but get "
                             << args.size();
  TXSessionOptions opt = DEFAULT_SESSION_OPTIONS;
  if (args.size() == 1) {
    opt.name = args[0].As<Unicode>().encode();
  }
  return new TXSession(opt);
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.FreeTXSessionHandle").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[FreeTXSessionHandle] Expect 1 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  delete static_cast<TXSession*>(handle);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSetDevice").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[TXSessionSetDevice] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  int64_t device = args[1].As<int64_t>();
  auto sess = static_cast<TXSession*>(handle);
  sess->SetDevice(device);
  return None;
});
MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionTrace").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionTrace] Expect 2 or more arguments but get "
                             << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  std::vector<const Symbol*> tmp;
  for (int i = 1; i < args.size(); ++i) {
    tmp.push_back(static_cast<Symbol*>(args[i].As<void*>()));
  }
  sess->Trace(tmp);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSetSchedulingThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() >= 1 || args.size() <= 3)
          << "[TXSessionSetSchedulingThreads] Expect 1 ~ 3 arguments but get " << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      switch (args.size()) {
        case 1: {
          sess->SetSchedulingThreads();
        } break;
        case 2: {
          int64_t thread_num = args[1].As<int64_t>();
          sess->SetSchedulingThreads(thread_num);
        } break;
        case 3: {
          int64_t thread_num = args[1].As<int64_t>();
          bool share = args[2].As<bool>();
          sess->SetSchedulingThreads(thread_num, share);
        } break;
        default: {
          // unreachable code
        } break;
      }
      return None;
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionGetSchedulingThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() == 1) << "[TXSessionGetSchedulingThreads] Expect 1  arguments but get "
                                << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      return sess ? sess->GetSchedulingThreads() : 0;
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSetOpParallelismThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() >= 1 || args.size() <= 3)
          << "[TXSessionSetOpParallelismThreads] Expect 1 ~ 3 arguments but get " << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      switch (args.size()) {
        case 1: {
          sess->SetOpParallelismThreads();
        } break;
        case 2: {
          int64_t thread_num = args[1].As<int64_t>();
          sess->SetOpParallelismThreads(thread_num);
        } break;
        case 3: {
          int64_t thread_num = args[1].As<int64_t>();
          bool share = args[2].As<bool>();
          sess->SetOpParallelismThreads(thread_num, share);
        } break;
        default: {
          // unreachable code
        } break;
      }
      return None;
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionGetOpParallelismThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() == 1) << "[TXSessionGetOpParallelismThreads] Expect 1  arguments but get "
                                << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      return sess ? sess->GetOpParallelismThreads() : 0;
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSetOpComputeThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() >= 1 || args.size() <= 3)
          << "[TXSessionSetOpComputeThreads] Expect 1 ~ 3 arguments but get " << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      switch (args.size()) {
        case 1: {
          sess->SetOpComputeThreads();
        } break;
        case 2: {
          int64_t thread_num = args[1].As<int64_t>();
          sess->SetOpComputeThreads(thread_num);
        } break;
        case 3: {
          int64_t thread_num = args[1].As<int64_t>();
          bool share = args[2].As<bool>();
          sess->SetOpComputeThreads(thread_num);
        } break;
        default: {
          // unreachable code
        } break;
      }
      return None;
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionGetOpComputeThreads")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK(args.size() == 1) << "[TXSessionGetOpComputeThreads] Expect 1 arguments but get "
                                << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      if (sess && sess->GetComputeThreadPool()) {
        return int64_t(sess->GetComputeThreadPool()->GetThreadsNum());
      }
      return int64_t(0);
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSave").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[TXSessionSave] Expect 3 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  Unicode folder = args[1].As<Unicode>();
  Unicode name = args[2].As<Unicode>();
  auto sess = static_cast<TXSession*>(handle);
  sess->Save(folder.encode(), name.encode());
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionRun").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionRun] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Dict feed_dict = args[1].As<Dict>();
  std::unordered_map<std::string, RTValue> feed_dict_v2;
  for (auto kv : feed_dict.items()) {
    feed_dict_v2.emplace(kv.first.As<String>(), kv.second);
  }
  auto result = sess->Run(feed_dict_v2);
  List result_v2;
  for (auto& item : result) {
    result_v2.append(std::move(item.second));
  }
  return result_v2;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionRunWithMeta").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionRunWithMeta] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Dict feed_dict = args[1].As<Dict>();
  std::unordered_map<std::string, RTValue> feed_dict_v2;
  for (auto kv : feed_dict.items()) {
    feed_dict_v2.emplace(kv.first.As<String>(), kv.second);
  }
  Dict meta_info;
  TXSessionRunMeta meta;
  auto result = sess->Run(feed_dict_v2, &meta);
  List result_v2;
  for (auto& item : result) {
    result_v2.append(std::move(item.second));
  }
  List ops_meta;
  for (auto& step_st : meta.step_stats) {
    Dict op_meta;
    op_meta[U"op"] = step_st.op.decode();
    op_meta[U"op_cls"] = step_st.op_cls.decode();
    op_meta[U"start"] = step_st.time_line.stamp_start;
    op_meta[U"end"] = step_st.time_line.stamp_end;
    op_meta[U"inputs"] = step_st.inputs;
    op_meta[U"output"] = step_st.output;
    op_meta[U"attributes"] = step_st.attributes;
    ops_meta.append(std::move(op_meta));
  }
  meta_info[U"start"] = meta.time_line.stamp_start;
  meta_info[U"end"] = meta.time_line.stamp_end;
  meta_info[U"ops"] = ops_meta;
  return Tuple::dynamic(std::move(result_v2), std::move(meta_info));
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionGetNestedOpAttributesByName")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 3)
          << "[TXSessionGetNestedOpAttributesByName] Expect 3 arguments but get " << args.size();
      void* handle = args[0].As<void*>();
      auto sess = static_cast<TXSession*>(handle);
      string_view op_cls = args[1].As<string_view>();
      string_view op_name = args[2].As<string_view>();
      auto op_ptr = sess->FindOp(op_cls, op_name);
      if (op_ptr) {
        return TXSession::GetNestedOpAttributes(op_ptr.get());
      } else {
        return None;
      }
    });

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionWarmup").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionWarmup] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Dict feed_dict = args[1].As<Dict>();
  std::unordered_map<std::string, RTValue> feed_dict_v2;
  for (auto kv : feed_dict.items()) {
    feed_dict_v2.emplace(kv.first.As<String>(), kv.second);
  }
  auto result = sess->Warmup(feed_dict_v2);
  List result_v2;
  for (auto& item : result) {
    result_v2.append(std::move(item.second));
  }
  return result_v2;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.LoadTXSession").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[LoadTXSession] Expect 3 arguments but get " << args.size();
  Unicode folder = args[0].As<Unicode>();
  Unicode name = args[1].As<Unicode>();
  int64_t device = -1;
  switch (args[2].type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      auto ctx = NDArrayHelper::GetDevice(args[2].AsNoCheck<Unicode>());
      MXCHECK(ctx.device_type == kDLCPU || ctx.device_type == kDLCUDA);
      if (ctx.device_type == kDLCUDA) {
        device = ctx.device_id;
      }
    } break;
    case TypeIndex::kRuntimeString: {
      auto ctx = NDArrayHelper::GetDevice(args[2].AsNoCheck<String>().decode());
      MXCHECK(ctx.device_type == kDLCPU || ctx.device_type == kDLCUDA);
      if (ctx.device_type == kDLCUDA) {
        device = ctx.device_id;
      }
    } break;
    case TypeIndex::kRuntimeInteger: {
      device = args[2].AsNoCheck<int64_t>();
    } break;
    default: {
      MXTHROW << "expect device is int or str type, but get " << args[2];
    } break;
  }
  std::unique_ptr<TXSession> ptr = TXSession::Load(folder.encode(), name.encode(), device);
  return ptr.release();
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionGetAttr").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionRun] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Unicode key = args[1].As<Unicode>();
  return sess->GetAttr(key.encode());
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionSetAttr").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 3) << "[TXSessionRun] Expect 3 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Unicode key = args[1].As<Unicode>();
  RTValue val = args[2].As<RTValue>();
  sess->SetAttr(key.encode(), val);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.TXSessionHasAttr").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[TXSessionRun] Expect 2 arguments but get " << args.size();
  void* handle = args[0].As<void*>();
  auto sess = static_cast<TXSession*>(handle);
  Unicode key = args[1].As<Unicode>();
  return sess->HasAttr(key.encode());
});

extern RTValue ParallelMap(const UserDataRef& func, const Any& inputs, void* session_handle);
extern RTValue ParallelStarMap(const UserDataRef& func, const Any& inputs, void* session_handle);
extern RTValue ApplyAsync(const UserDataRef& func, const PyArgs& inputs, void* session_handle);

MATXSCRIPT_REGISTER_GLOBAL("pipeline.ParallelMap").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[ParallelMap] Expect 3 arguments but get " << args.size();
  auto func = args[0].As<UserDataRef>();
  auto* sess = args[2].As<void*>();
  return ParallelMap(func, args[1], sess);
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.ParallelStarMap").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[ParallelStarMap] Expect 3 arguments but get " << args.size();
  auto func = args[0].As<UserDataRef>();
  auto* sess = args[2].As<void*>();
  return ParallelStarMap(func, args[1], sess);
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.ApplyAsync").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[ApplyAsync] Expect 2 or more arguments but get " << args.size();
  auto func = args[0].As<UserDataRef>();
  auto* sess = args[args.size() - 1].As<void*>();
  return ApplyAsync(func, PyArgs(args.begin() + 1, args.size() - 2), sess);
});

MATXSCRIPT_REGISTER_GLOBAL("pipeline.PythonBaseOp_UpdatePassOpOptions")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK_GE(args.size(), 2) << "[UpdatePythonBaseOp] Expect 2 or more arguments but get "
                                 << args.size();
      UserDataRef ud = args[0].As<UserDataRef>();
      OpKernelPtr op_ptr = check_get_op_kernel(ud);
      MXCHECK(op_ptr->ClassName() == "PythonBaseOp") << "internal error";
      auto py_op = std::static_pointer_cast<PythonBaseOp>(op_ptr);
      auto new_op_options = args[1].As<Dict>();
      auto items = new_op_options.items();
      for (auto kv : items) {
        py_op->pass_op_options.set_item(kv.first, kv.second);
      }
      return None;
    });

}  // namespace runtime
}  // namespace matxscript
