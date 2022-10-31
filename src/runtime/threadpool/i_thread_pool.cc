// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include <matxscript/runtime/threadpool/i_thread_pool.h>
#include <exception>

namespace matxscript {
namespace runtime {
namespace internal {

void IThreadPool::WaitBulk(std::vector<IRunnablePtr>& runners) {
  std::exception_ptr eptr;
  for (auto& runner : runners) {
    try {
      runner->Wait();
    } catch (...) {
      if (!eptr) {
        eptr = std::current_exception();
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript