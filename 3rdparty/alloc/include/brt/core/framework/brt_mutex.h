// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#pragma once
#ifdef _WIN32
#define NOMINMAX  // disable Win's min/max macro
#include <Windows.h>
#include <mutex>
namespace brt {
// Q: Why BrtMutex is better than std::mutex
// A: BrtMutex supports static initialization but std::mutex doesn't. Static initialization helps us
// prevent the "static initialization order problem".

// Q: Why std::mutex can't make it?
// A: VC runtime has to support Windows XP at ABI level. But we don't have such requirement.

// Q: Is BrtMutex faster than std::mutex?
// A: Sure

class BrtMutex {
 private:
  SRWLOCK data_ = SRWLOCK_INIT;

 public:
  BrtMutex() = default;
  // SRW locks do not need to be explicitly destroyed.
  ~BrtMutex() = default;
  BrtMutex(const BrtMutex&) = delete;
  BrtMutex& operator=(const BrtMutex&) = delete;
  void lock() {
    AcquireSRWLockExclusive(native_handle());
  }
  bool try_lock() noexcept {
    return TryAcquireSRWLockExclusive(native_handle()) == TRUE;
  }
  void unlock() noexcept {
    ReleaseSRWLockExclusive(native_handle());
  }
  using native_handle_type = SRWLOCK*;

  __forceinline native_handle_type native_handle() {
    return &data_;
  }
};

class BrtCondVar {
  CONDITION_VARIABLE native_cv_object = CONDITION_VARIABLE_INIT;

 public:
  BrtCondVar() noexcept = default;
  ~BrtCondVar() = default;

  BrtCondVar(const BrtCondVar&) = delete;
  BrtCondVar& operator=(const BrtCondVar&) = delete;

  void notify_one() noexcept {
    WakeConditionVariable(&native_cv_object);
  }
  void notify_all() noexcept {
    WakeAllConditionVariable(&native_cv_object);
  }

  void wait(std::unique_lock<BrtMutex>& lk) {
    if (SleepConditionVariableSRW(&native_cv_object, lk.mutex()->native_handle(), INFINITE, 0) !=
        TRUE) {
      std::terminate();
    }
  }
  template <class _Predicate>
  void wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the
   * method returns cv_status::no_timeout.
   * @param cond_mutex A unique_lock<BrtMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread
   * wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise,
   * the method returns cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                          const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = CONDITION_VARIABLE*;

  native_handle_type native_handle() {
    return &native_cv_object;
  }

 private:
  void timed_wait_impl(
      std::unique_lock<BrtMutex>& __lk,
      std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

template <class _Predicate>
void BrtCondVar::wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred) {
  while (!__pred())
    wait(__lk);
}

template <class Rep, class Period>
std::cv_status BrtCondVar::wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  // TODO: is it possible to use nsync_from_time_point_ ?
  using namespace std::chrono;
  if (rel_time <= duration<Rep, Period>::zero())
    return std::cv_status::timeout;
  using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano>>;
  using SystemTimePoint = time_point<system_clock, nanoseconds>;
  SystemTimePointFloat max_time = SystemTimePoint::max();
  steady_clock::time_point steady_now = steady_clock::now();
  system_clock::time_point system_now = system_clock::now();
  if (max_time - rel_time > system_now) {
    nanoseconds remain = duration_cast<nanoseconds>(rel_time);
    if (remain < rel_time)
      ++remain;
    timed_wait_impl(cond_mutex, system_now + remain);
  } else
    timed_wait_impl(cond_mutex, SystemTimePoint::max());
  return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout
                                                     : std::cv_status::timeout;
}
}  // namespace brt
#else

#if USE_NSYNC

#include <condition_variable>  //for cv_status
#include <mutex>               //for unique_lock
#include "nsync.h"
namespace brt {

class BrtMutex {
  nsync::nsync_mu data_ = NSYNC_MU_INIT;

 public:
  BrtMutex() = default;
  ~BrtMutex() = default;
  BrtMutex(const BrtMutex&) = delete;
  BrtMutex& operator=(const BrtMutex&) = delete;

  void lock() {
    nsync::nsync_mu_lock(&data_);
  }
  bool try_lock() noexcept {
    return nsync::nsync_mu_trylock(&data_) == 0;
  }
  void unlock() noexcept {
    nsync::nsync_mu_unlock(&data_);
  }

  using native_handle_type = nsync::nsync_mu*;
  native_handle_type native_handle() {
    return &data_;
  }
};

class BrtCondVar {
  nsync::nsync_cv native_cv_object = NSYNC_CV_INIT;

 public:
  BrtCondVar() noexcept = default;

  ~BrtCondVar() = default;
  BrtCondVar(const BrtCondVar&) = delete;
  BrtCondVar& operator=(const BrtCondVar&) = delete;

  void notify_one() noexcept {
    nsync::nsync_cv_signal(&native_cv_object);
  }
  void notify_all() noexcept {
    nsync::nsync_cv_broadcast(&native_cv_object);
  }

  void wait(std::unique_lock<BrtMutex>& lk);
  template <class _Predicate>
  void wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the
   * method returns cv_status::no_timeout.
   * @param cond_mutex A unique_lock<BrtMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread
   * wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise,
   * the method returns cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                          const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = nsync::nsync_cv*;
  native_handle_type native_handle() {
    return &native_cv_object;
  }

 private:
  void timed_wait_impl(
      std::unique_lock<BrtMutex>& __lk,
      std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

template <class _Predicate>
void BrtCondVar::wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred) {
  while (!__pred())
    wait(__lk);
}

template <class Rep, class Period>
std::cv_status BrtCondVar::wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  // TODO: is it possible to use nsync_from_time_point_ ?
  using namespace std::chrono;
  if (rel_time <= duration<Rep, Period>::zero())
    return std::cv_status::timeout;
  using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano>>;
  using SystemTimePoint = time_point<system_clock, nanoseconds>;
  SystemTimePointFloat max_time = SystemTimePoint::max();
  steady_clock::time_point steady_now = steady_clock::now();
  system_clock::time_point system_now = system_clock::now();
  if (max_time - rel_time > system_now) {
    nanoseconds remain = duration_cast<nanoseconds>(rel_time);
    if (remain < rel_time)
      ++remain;
    timed_wait_impl(cond_mutex, system_now + remain);
  } else
    timed_wait_impl(cond_mutex, SystemTimePoint::max());
  return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout
                                                     : std::cv_status::timeout;
}
};  // namespace brt
#else

#include <condition_variable>
#include <mutex>
namespace brt {

class BrtMutex {
  std::mutex mtx;

 public:
  BrtMutex() = default;
  ~BrtMutex() = default;
  BrtMutex(const BrtMutex&) = delete;
  BrtMutex& operator=(const BrtMutex&) = delete;

  void lock() {
    mtx.lock();
  }
  bool try_lock() noexcept {
    return mtx.try_lock();
  }
  void unlock() noexcept {
    mtx.unlock();
  }

  using native_handle_type = std::mutex::native_handle_type;
  native_handle_type native_handle() {
    return mtx.native_handle();
  }
};

class BrtCondVar {
  std::condition_variable cv;

 public:
  BrtCondVar() noexcept = default;

  ~BrtCondVar() = default;
  BrtCondVar(const BrtCondVar&) = delete;
  BrtCondVar& operator=(const BrtCondVar&) = delete;

  void notify_one() noexcept {
    cv.notify_one();
  }
  void notify_all() noexcept {
    cv.notify_all();
  }

  void wait(std::unique_lock<BrtMutex>& lk);
  template <class _Predicate>
  void wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the
   * method returns cv_status::no_timeout.
   * @param cond_mutex A unique_lock<BrtMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread
   * wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise,
   * the method returns cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                          const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = std::condition_variable::native_handle_type;
  native_handle_type native_handle() {
    return cv.native_handle();
  }

 private:
  void timed_wait_impl(
      std::unique_lock<BrtMutex>& __lk,
      std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

template <class _Predicate>
void BrtCondVar::wait(std::unique_lock<BrtMutex>& __lk, _Predicate __pred) {
  while (!__pred())
    wait(__lk);
}

template <class Rep, class Period>
std::cv_status BrtCondVar::wait_for(std::unique_lock<BrtMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  return cv.wait_for(cond_mutex, rel_time);
}
};  // namespace brt

#endif
#endif
