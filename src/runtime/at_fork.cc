// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file is inspired by folly AtFork.
 * https://github.com/facebook/folly/blob/main/folly/system/AtFork.cpp
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
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

#include <matxscript/runtime/at_fork.h>

#include <sstream>

#ifndef MATXSCRIPT_HAVE_PTHREAD_ATFORK
#define MATXSCRIPT_HAVE_PTHREAD_ATFORK (!(defined(_WIN32) || defined(WIN32)))
#endif

#ifdef MATXSCRIPT_HAVE_PTHREAD_ATFORK
#include <pthread.h>
#endif

namespace matxscript {
namespace runtime {
namespace internal {

namespace {
//  AtForkList
//
//  The list data structure used internally in AtFork's internal singleton.
//
//  Useful for AtFork, but may not be useful more broadly.
class AtForkList {
 public:
  //  prepare
  //
  //  Acquires the mutex. Performs trial passes in a loop until a trial pass
  //  succeeds. A trial pass invokes each prepare handler in reverse order of
  //  insertion, failing the pass and cleaning up if any handler returns false.
  //  Cleanup entails invoking parent handlers in reverse order, up to but not
  //  including the parent handler corresponding to the failed prepare handler.
  void prepare() noexcept;

  //  parent
  //
  //  Invokes parent handlers in order of insertion. Releases the mutex.
  void parent() noexcept;

  //  child
  //
  //  Invokes child handlers in order of insertion. Releases the mutex.
  void child() noexcept;

  //  append
  //
  //  While holding the mutex, inserts a set of handlers to the end of the list.
  //
  //  If handle is not nullptr, the set of handlers is indexed by handle and may
  //  be found by members remove() and contain().
  void append(  //
      void const* handle,
      std::function<bool()> prepare,
      std::function<void()> parent,
      std::function<void()> child);

  //  remove
  //
  //  While holding the mutex, removes a set of handlers found by handle, if not
  //  null.
  void remove(  //
      void const* handle);

  //  contains
  //
  //  While holding the mutex, finds a set of handlers found by handle, if not
  //  null, returning true if found and false otherwise.
  bool contains(  //
      void const* handle);

 private:
  struct Task {
    void const* handle;
    std::function<bool()> prepare;
    std::function<void()> parent;
    std::function<void()> child;
  };

  std::mutex mutex;
  std::list<Task> tasks;
  std::map<void const*, std::list<Task>::iterator> index;
};

void AtForkList::prepare() noexcept {
  mutex.lock();
  while (true) {
    auto task = tasks.rbegin();
    for (; task != tasks.rend(); ++task) {
      if (auto& f = task->prepare) {
        if (!f()) {
          break;
        }
      }
    }
    if (task == tasks.rend()) {
      return;
    }
    for (auto untask = tasks.rbegin(); untask != task; ++untask) {
      if (auto& f = untask->parent) {
        f();
      }
    }
  }
}

void AtForkList::parent() noexcept {
  for (auto& task : tasks) {
    if (auto& f = task.parent) {
      f();
    }
  }
  mutex.unlock();
}

void AtForkList::child() noexcept {
  for (auto& task : tasks) {
    if (auto& f = task.child) {
      f();
    }
  }
  mutex.unlock();
}

void AtForkList::append(void const* handle,
                        std::function<bool()> prepare,
                        std::function<void()> parent,
                        std::function<void()> child) {
  std::unique_lock<std::mutex> lg{mutex};
  if (handle && index.count(handle)) {
    std::stringstream os;
    os << __FILE__ << ":" << __LINE__ << "at-fork: append: duplicate";
    throw std::invalid_argument(os.str());
  }
  auto task = Task{handle, std::move(prepare), std::move(parent), std::move(child)};
  auto inserted = tasks.insert(tasks.end(), std::move(task));
  if (handle) {
    index.emplace(handle, inserted);
  }
}

void AtForkList::remove(void const* handle) {
  if (!handle) {
    return;
  }
  std::unique_lock<std::mutex> lg{mutex};
  auto i1 = index.find(handle);
  if (i1 == index.end()) {
    std::stringstream os;
    os << __FILE__ << ":" << __LINE__ << "at-fork: remove: missing";
    throw std::out_of_range(os.str());
  }
  auto i2 = i1->second;
  index.erase(i1);
  tasks.erase(i2);
}

bool AtForkList::contains(  //
    void const* handle) {
  if (!handle) {
    return false;
  }
  std::unique_lock<std::mutex> lg{mutex};
  return index.count(handle) != 0;
}

struct SkipAtForkHandlers {
  static thread_local bool value;

  struct Guard {
    bool saved = value;
    Guard() {
      value = true;
    }
    ~Guard() {
      value = saved;
    }
  };
};
thread_local bool SkipAtForkHandlers::value;

void invoke_pthread_atfork(void (*prepare)(), void (*parent)(), void (*child)()) {
  int ret = 0;
#ifdef MATXSCRIPT_HAVE_PTHREAD_ATFORK  // if no pthread_atfork, probably no fork either
  ret = pthread_atfork(prepare, parent, child);
#endif
  if (ret != 0) {
    std::stringstream os;
    os << __FILE__ << ":" << __LINE__ << "pthread_atfork failed";
    throw std::system_error(ret, std::generic_category(), os.str());
  }
}

struct AtForkListSingleton {
  static void init() {
    static int reg = (get(), invoke_pthread_atfork(prepare, parent, child), 0);
    (void)reg;
  }

  static AtForkList& get() {
    static auto& instance = *new AtForkList();
    return instance;
  }

  static void prepare() {
    if (!SkipAtForkHandlers::value) {
      get().prepare();
    }
  }

  static void parent() {
    if (!SkipAtForkHandlers::value) {
      get().parent();
    }
  }

  static void child() {
    if (!SkipAtForkHandlers::value) {
      // if we fork a multithreaded process
      // some of the TSAN mutexes might be locked
      // so we just enable ignores for everything
      // while handling the child callbacks
      // This might still be an issue if we do not exec right away
      // TODO: fix: annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
      get().child();
    }
  }
};

}  // namespace

void AtFork::RegisterHandler(void const* handle,
                             std::function<bool()> prepare,
                             std::function<void()> parent,
                             std::function<void()> child) {
  AtForkListSingleton::init();
  auto& list = AtForkListSingleton::get();
  list.append(handle, std::move(prepare), std::move(parent), std::move(child));
}

void AtFork::UnregisterHandler(void const* handle) {
  auto& list = AtForkListSingleton::get();
  list.remove(handle);
}

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
