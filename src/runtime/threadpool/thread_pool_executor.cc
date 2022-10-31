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
#include <matxscript/runtime/threadpool/thread_pool_executor.h>

#include <matxscript/runtime/future_wrap.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/threadpool/lock_based_thread_pool.h>
#include <matxscript/runtime/threadpool/lock_free_thread_pool.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace runtime {

template <typename RunnableType, bool UnpackArgs = false>
class ParallelForTask : public RunnableType {
 public:
  ParallelForTask(const UserDataRef& op, const Any* input_first, RTValue* output_first, int64_t len)
      : op_(&op),
        input_first_(input_first),
        input_last_(input_first + len),
        output_first_(output_first) {
  }

  void RunImpl() override {
    while (input_first_ != input_last_) {
      if (UnpackArgs) {
        switch (input_first_->type_code()) {
          case TypeIndex::kRuntimeList: {
            auto args = input_first_->template AsObjectRefNoCheck<List>();
            *output_first_ = op_->generic_call(PyArgs(args.data(), args.size()));
          } break;
          case TypeIndex::kRuntimeTuple: {
            auto args = input_first_->template AsObjectRefNoCheck<Tuple>();
            *output_first_ = op_->generic_call(PyArgs(args.begin(), args.size()));
          } break;
          case TypeIndex::kRuntimeFTList: {
            auto num_args = kernel_object___len__(*input_first_);
            Iterator iterable = Kernel_Iterable::make(*input_first_);
            std::vector<RTValue> args;
            args.reserve(num_args);
            bool has_next = iterable.HasNext();
            while (has_next) {
              args.emplace_back(iterable.Next(&has_next));
            }
            *output_first_ = op_->generic_call(PyArgs(args.data(), args.size()));
          } break;
          default: {
            MXTHROW << "matx.pstarmap(f, iterable) expect iterable[i] is list or tuple, but get "
                    << input_first_->type_name();
          } break;
        }
      } else {
        *output_first_ = op_->generic_call(PyArgs(input_first_, 1));
      }
      ++input_first_;
      ++output_first_;
    }
  }

 private:
  const UserDataRef* op_;
  const Any* input_first_;
  const Any* input_last_;
  RTValue* output_first_;
};

template <typename RunnableType>
struct AsyncTask : public RunnableType {
  UserDataRef closure;
  std::vector<RTValue> args;
  RTValue result;
  AsyncTask(UserDataRef closure, std::vector<RTValue> args)
      : closure(std::move(closure)), args(std::move(args)) {
  }

  void RunImpl() override {
    result = closure.generic_call(PyArgs(args.data(), args.size()));
  }
};

ThreadPoolExecutor::ThreadPoolExecutor(const std::shared_ptr<internal::IThreadPool>& pool,
                                       bool lock_free)
    : lock_free_(lock_free), thread_num_(pool->GetThreadsNum()), pool_(pool) {
  auto t_ids = pool->GetThreadIds();
  for (auto& id : t_ids) {
    pool_thread_ids_.emplace(id);
  }
}

void ThreadPoolExecutor::ParallelForImpl(const UserDataRef& op,
                                         const Any* inputs_begin,
                                         const Any* inputs_end,
                                         int64_t expt_num_threads,
                                         int64_t group_size,
                                         RTValue* outputs_begin,
                                         bool unpack_args) {
  int64_t input_size = inputs_end - inputs_begin;
  if (expt_num_threads <= 0) {
    expt_num_threads = thread_num_ + 1;
  }
  if (group_size <= 0) {
    group_size = 1;
  }
  MXCHECK(input_size % group_size == 0) << "Expect the number of tasks to be a multiple of "
                                        << group_size << ", but get " << input_size << "";
  int64_t num_group = input_size / group_size;

  int64_t step_r = group_size * ((num_group + expt_num_threads - 1) / expt_num_threads);
  int64_t step_l = step_r - group_size;

  int64_t pos = 0;
  int64_t step = step_r;
  bool need_change = true;
  std::vector<internal::IRunnablePtr> tasks;
  tasks.reserve(expt_num_threads);
  for (int64_t i = 0; i < expt_num_threads && pos < input_size; ++i) {
    if (need_change && step_l != 0 && pos + step_l * (expt_num_threads - i) == input_size) {
      step = step_l;
      need_change = false;
    }
    if (lock_free_) {
      if (unpack_args) {
        auto task = std::make_shared<ParallelForTask<internal::LockFreeRunnable, true>>(
            op, inputs_begin + pos, outputs_begin + pos, step);
        tasks.push_back(std::static_pointer_cast<internal::IRunnable>(task));
      } else {
        auto task = std::make_shared<ParallelForTask<internal::LockFreeRunnable, false>>(
            op, inputs_begin + pos, outputs_begin + pos, step);
        tasks.push_back(std::static_pointer_cast<internal::IRunnable>(task));
      }
    } else {
      if (unpack_args) {
        auto task = std::make_shared<ParallelForTask<internal::LockBasedRunnable, true>>(
            op, inputs_begin + pos, outputs_begin + pos, step);
        tasks.push_back(std::static_pointer_cast<internal::IRunnable>(task));
      } else {
        auto task = std::make_shared<ParallelForTask<internal::LockBasedRunnable, false>>(
            op, inputs_begin + pos, outputs_begin + pos, step);
        tasks.push_back(std::static_pointer_cast<internal::IRunnable>(task));
      }
    }
    pos += step;
  }

  auto cur_tid = std::this_thread::get_id();
  if (pool_thread_ids_.find(cur_tid) != pool_thread_ids_.end()) {
    // fix nested pmap
    for (auto& task : tasks) {
      task->Run();
    }
  } else {
    size_t task_size = tasks.size();
    if (task_size > 1) {
      size_t seq = serial_.fetch_add(task_size - 1, std::memory_order_relaxed);
      for (size_t i = 1; i < tasks.size(); ++i) {
        pool_->Enqueue(tasks[i], seq + i - 1);
      }
    }
    internal::IRunnablePtr& first_task = tasks[0];

    first_task->Run();
  }

  internal::IThreadPool::WaitBulk(tasks);
}

List ThreadPoolExecutor::ParallelFor(const UserDataRef& op,
                                     const List& inputs,
                                     int64_t expt_num_threads,
                                     int64_t group_size) {
  int64_t input_size = inputs.size();
  List outputs(input_size, None);

  if (input_size == 0) {
    return outputs;
  }
  auto* inputs_data = inputs.data();
  ParallelForImpl(op,
                  inputs_data,
                  inputs_data + input_size,
                  expt_num_threads,
                  group_size,
                  outputs.data(),
                  false);
  return outputs;
}

List ThreadPoolExecutor::ParallelFor(const UserDataRef& op, const List& inputs) {
  return ParallelFor(op, inputs, thread_num_ + 1, 1);
}

Tuple ThreadPoolExecutor::ParallelFor(const UserDataRef& op,
                                      const Tuple& inputs,
                                      int64_t expt_num_threads,
                                      int64_t group_size) {
  int64_t input_size = inputs.size();
  auto output_node = TupleNode::MakeNones(input_size);

  if (input_size == 0) {
    return Tuple(std::move(output_node));
  }
  auto* inputs_data = inputs.begin();
  ParallelForImpl(op,
                  inputs_data,
                  inputs_data + input_size,
                  expt_num_threads,
                  group_size,
                  output_node->data(),
                  false);
  return Tuple(std::move(output_node));
}

Tuple ThreadPoolExecutor::ParallelFor(const UserDataRef& op, const Tuple& inputs) {
  return ParallelFor(op, inputs, thread_num_ + 1, 1);
}

List ThreadPoolExecutor::ParallelStarMap(const UserDataRef& op,
                                         const List& inputs,
                                         int64_t expt_num_threads,
                                         int64_t group_size) {
  int64_t input_size = inputs.size();
  List outputs(input_size, None);

  if (input_size == 0) {
    return outputs;
  }
  auto* inputs_data = inputs.data();
  ParallelForImpl(op,
                  inputs_data,
                  inputs_data + input_size,
                  expt_num_threads,
                  group_size,
                  outputs.data(),
                  true);
  return outputs;
}

List ThreadPoolExecutor::ParallelStarMap(const UserDataRef& op, const List& inputs) {
  return ParallelStarMap(op, inputs, thread_num_ + 1, 1);
}

Tuple ThreadPoolExecutor::ParallelStarMap(const UserDataRef& op,
                                          const Tuple& inputs,
                                          int64_t expt_num_threads,
                                          int64_t group_size) {
  int64_t input_size = inputs.size();
  auto output_node = TupleNode::MakeNones(input_size);

  if (input_size == 0) {
    return Tuple(std::move(output_node));
  }
  auto* inputs_data = inputs.begin();
  ParallelForImpl(op,
                  inputs_data,
                  inputs_data + input_size,
                  expt_num_threads,
                  group_size,
                  output_node->data(),
                  true);
  return Tuple(std::move(output_node));
}

Tuple ThreadPoolExecutor::ParallelStarMap(const UserDataRef& op, const Tuple& inputs) {
  return ParallelStarMap(op, inputs, thread_num_ + 1, 1);
}

RTValue ThreadPoolExecutor::ApplyAsync(const UserDataRef& callable, const PyArgs& args) {
  auto cur_tid = std::this_thread::get_id();
  if (pool_thread_ids_.find(cur_tid) != pool_thread_ids_.end()) {
    // fix nested apply_async
    auto result = callable->generic_call(args);
    return Future::make_future_udref([r = std::move(result)]() mutable { return r; });
  }
  size_t seq = serial_.fetch_add(1, std::memory_order_relaxed);
  std::vector<RTValue> args_holder;
  args_holder.reserve(args.size());
  for (auto i = 0; i < args.size(); ++i) {
    args_holder.emplace_back(args[i].As<RTValue>());
  }
  if (lock_free_) {
    auto closure_task =
        std::make_shared<AsyncTask<internal::LockFreeRunnable>>(callable, std::move(args_holder));
    auto my_closure_task = std::static_pointer_cast<internal::IRunnable>(closure_task);
    pool_->Enqueue(my_closure_task, seq);
    return Future::make_future_udref([closure_task]() {
      closure_task->Wait();
      return closure_task->result;
    });
  } else {
    auto closure_task =
        std::make_shared<AsyncTask<internal::LockBasedRunnable>>(callable, std::move(args_holder));
    auto my_closure_task = std::static_pointer_cast<internal::IRunnable>(closure_task);
    pool_->Enqueue(my_closure_task, seq);
    return Future::make_future_udref([closure_task]() {
      closure_task->Wait();
      return closure_task->result;
    });
  }
}

RTValue ThreadPoolExecutor::Submit(PyArgs args) {
  if (args.size() < 1) {
    THROW_PY_TypeError("[ThreadPoolExecutor][func: Submit] Expect 1 or more arguments but get ",
                       args.size());
  }
  if (!args[0].IsObjectRef<UserDataRef>()) {
    THROW_PY_TypeError(
        "[ThreadPoolExecutor][func: Submit] Expect the first argument is a callable object, but get ",
        args[0].type_name());
  }
  auto callable = args[0].AsObjectViewNoCheck<UserDataRef>();
  return this->ApplyAsync(callable.data(), PyArgs(args.begin() + 1, args.size() - 1));
}

MATX_REGISTER_NATIVE_OBJECT(ThreadPoolExecutor)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 2 || args.size() == 3)
          << "[ThreadPoolExecutor] Expect 2 or 3 arguments but get " << args.size();
      int pool_size = args[0].As<int64_t>();
      bool lock_free = args[1].As<bool>();
      int64_t intervals_ns = 1;
      if (args.size() == 3) {
        intervals_ns = args[2].As<int64_t>();
      }
      auto pool = lock_free
                      ? std::unique_ptr<internal::IThreadPool>(new internal::SPSCLockFreeThreadPool(
                            pool_size, "matx.ThreadPool", intervals_ns))
                      : std::unique_ptr<internal::IThreadPool>(
                            new internal::LockBasedThreadPool(pool_size, "matx.ThreadPool"));
      return std::make_shared<ThreadPoolExecutor>(std::move(pool), lock_free);
    })
    .RegisterFunction(
        "ParallelFor",
        [](void* self, PyArgs args) -> RTValue {
          MXCHECK(args.size() >= 2 && args.size() <= 4)
              << "[ThreadPoolExecutor][func: ParallelFor] Expect 2-4 arguments but get "
              << args.size();
          UserDataRef op = args[0].As<UserDataRef>();
          List inputs = args[1].As<List>();
          int64_t expt_num_threads = -1;
          int64_t group_size = 1;
          if (args.size() >= 3) {
            expt_num_threads = args[2].As<int64_t>();
          }
          if (args.size() >= 4) {
            group_size = args[3].As<int64_t>();
          }
          return reinterpret_cast<ThreadPoolExecutor*>(self)->ParallelFor(
              op, inputs, expt_num_threads, group_size);
        })
    .RegisterFunction("Submit",
                      [](void* self, PyArgs args) -> RTValue {
                        return reinterpret_cast<ThreadPoolExecutor*>(self)->Submit(args);
                      })
    .SetThreadSafety(false);

}  // namespace runtime
}  // namespace matxscript
