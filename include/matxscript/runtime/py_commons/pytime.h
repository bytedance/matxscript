// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pytime.h
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
#pragma once

#include <cstdint>
#include <ctime>
#ifdef MS_WINDOWS
#include <winsock2.h> /* struct timeval */
#else
#include <sys/time.h>
#endif

namespace matxscript {
namespace runtime {
namespace py_builtins {

/* _PyTime_t: Python timestamp with subsecond precision. It can be used to
   store a duration, and so indirectly a date (related to another date, like
   UNIX epoch). */
typedef int64_t _PyTime_t;
constexpr _PyTime_t _PyTime_MIN = INT64_MIN;
constexpr _PyTime_t _PyTime_MAX = INT64_MAX;

typedef enum {
  /* Round towards minus infinity (-inf).
     For example, used to read a clock. */
  _PyTime_ROUND_FLOOR = 0,
  /* Round towards infinity (+inf).
     For example, used for timeout to wait "at least" N seconds. */
  _PyTime_ROUND_CEILING = 1,
  /* Round to nearest with ties going to nearest even integer.
     For example, used to round from a Python float. */
  _PyTime_ROUND_HALF_EVEN = 2,
  /* Round away from zero
     For example, used for timeout. _PyTime_ROUND_CEILING rounds
     -1e-9 to 0 milliseconds which causes bpo-31786 issue.
     _PyTime_ROUND_UP rounds -1e-9 to -1 millisecond which keeps
     the timeout sign as expected. select.poll(timeout) must block
     for negative values." */
  _PyTime_ROUND_UP = 3,
  /* _PyTime_ROUND_TIMEOUT (an alias for _PyTime_ROUND_UP) should be
     used for timeouts. */
  _PyTime_ROUND_TIMEOUT = _PyTime_ROUND_UP
} _PyTime_round_t;

/* Convert a number of seconds, int or float, to time_t. */
int _PyTime_ObjectToTime_t(double obj, time_t* sec, _PyTime_round_t);

/* Convert a number of seconds, int or float, to a timeval structure.
   usec is in the range [0; 999999] and rounded towards zero.
   For example, -1.2 is converted to (-2, 800000). */
int _PyTime_ObjectToTimeval(double obj, time_t* sec, long* usec, _PyTime_round_t);

/* Convert a number of seconds, int or float, to a timespec structure.
   nsec is in the range [0; 999999999] and rounded towards zero.
   For example, -1.2 is converted to (-2, 800000000). */
int _PyTime_ObjectToTimespec(double obj, time_t* sec, long* nsec, _PyTime_round_t);

/* Create a timestamp from a number of seconds. */
_PyTime_t _PyTime_FromSeconds(int seconds);

/* Macro to create a timestamp from a number of seconds, no integer overflow.
   Only use the macro for small values, prefer _PyTime_FromSeconds(). */
inline constexpr _PyTime_t _PYTIME_FROMSECONDS(_PyTime_t seconds) {
  return ((_PyTime_t)(seconds) * (1000 * 1000 * 1000));
}

/* Create a timestamp from a number of nanoseconds. */
_PyTime_t _PyTime_FromNanoseconds(_PyTime_t ns);

/* Create a timestamp from nanoseconds (Python int). */
int _PyTime_FromNanosecondsObject(_PyTime_t* t, int64_t obj);

/* Convert a number of seconds (Python float or int) to a timetamp.
   Raise an exception and return -1 on error, return 0 on success. */
int _PyTime_FromSecondsObject(_PyTime_t* t, int64_t obj, _PyTime_round_t round);
int _PyTime_FromSecondsObject(_PyTime_t* t, double obj, _PyTime_round_t round);

/* Convert a number of milliseconds (Python float or int, 10^-3) to a timetamp.
   Raise an exception and return -1 on error, return 0 on success. */
int _PyTime_FromMillisecondsObject(_PyTime_t* t, int64_t obj, _PyTime_round_t round);
int _PyTime_FromMillisecondsObject(_PyTime_t* t, double obj, _PyTime_round_t round);

/* Convert a timestamp to a number of seconds as a C double. */
double _PyTime_AsSecondsDouble(_PyTime_t t);

/* Convert timestamp to a number of milliseconds (10^-3 seconds). */
_PyTime_t _PyTime_AsMilliseconds(_PyTime_t t, _PyTime_round_t round);

/* Convert timestamp to a number of microseconds (10^-6 seconds). */
_PyTime_t _PyTime_AsMicroseconds(_PyTime_t t, _PyTime_round_t round);

/* Create a timestamp from a timeval structure.
   Raise an exception and return -1 on overflow, return 0 on success. */
int _PyTime_FromTimeval(_PyTime_t* tp, timeval* tv);

/* Convert a timestamp to a timeval structure (microsecond resolution).
   tv_usec is always positive.
   Raise an exception and return -1 if the conversion overflowed,
   return 0 on success. */
int _PyTime_AsTimeval(_PyTime_t t, timeval* tv, _PyTime_round_t round);

/* Similar to _PyTime_AsTimeval(), but don't raise an exception on error. */
int _PyTime_AsTimeval_noraise(_PyTime_t t, timeval* tv, _PyTime_round_t round);

/* Convert a timestamp to a number of seconds (secs) and microseconds (us).
   us is always positive. This function is similar to _PyTime_AsTimeval()
   except that secs is always a time_t type, whereas the timeval structure
   uses a C long for tv_sec on Windows.
   Raise an exception and return -1 if the conversion overflowed,
   return 0 on success. */
int _PyTime_AsTimevalTime_t(_PyTime_t t, time_t* secs, int* us, _PyTime_round_t round);

/* Compute ticks * mul / div.
   The caller must ensure that ((div - 1) * mul) cannot overflow. */
_PyTime_t _PyTime_MulDiv(_PyTime_t ticks, _PyTime_t mul, _PyTime_t div);

/* Get the current time from the system clock.

   The function cannot fail. _PyTime_Init() ensures that the system clock
   works. */
_PyTime_t _PyTime_GetSystemClock(void);

/* Get the time of a monotonic clock, i.e. a clock that cannot go backwards.
   The clock is not affected by system clock updates. The reference point of
   the returned value is undefined, so that only the difference between the
   results of consecutive calls is valid.

   The function cannot fail. _PyTime_Init() ensures that a monotonic clock
   is available and works. */
_PyTime_t _PyTime_GetMonotonicClock(void);

/* Structure used by time.get_clock_info() */
typedef struct {
  const char* implementation;
  int monotonic;
  int adjustable;
  double resolution;
} _Py_clock_info_t;

/* Get the current time from the system clock.
 * Fill clock information if info is not NULL.
 * Raise an exception and return -1 on error, return 0 on success.
 */
int _PyTime_GetSystemClockWithInfo(_PyTime_t* t, _Py_clock_info_t* info);

/* Get the time of a monotonic clock, i.e. a clock that cannot go backwards.
   The clock is not affected by system clock updates. The reference point of
   the returned value is undefined, so that only the difference between the
   results of consecutive calls is valid.

   Fill info (if set) with information of the function used to get the time.

   Return 0 on success, raise an exception and return -1 on error. */
int _PyTime_GetMonotonicClockWithInfo(_PyTime_t* t, _Py_clock_info_t* info);

/* Initialize time.
   Return 0 on success, raise an exception and return -1 on error. */
int _PyTime_Init(void);

/* Converts a timestamp to the Gregorian time, using the local time zone.
   Return 0 on success, raise an exception and return -1 on error. */
int _PyTime_localtime(time_t t, struct tm* tm);

/* Converts a timestamp to the Gregorian time, assuming UTC.
   Return 0 on success, raise an exception and return -1 on error. */
int _PyTime_gmtime(time_t t, struct tm* tm);

/* Get the performance counter: clock with the highest available resolution to
   measure a short duration.

   The function cannot fail. _PyTime_Init() ensures that the system clock
   works. */
_PyTime_t _PyTime_GetPerfCounter(void);

/* Get the performance counter: clock with the highest available resolution to
   measure a short duration.

   Fill info (if set) with information of the function used to get the time.

   Return 0 on success, raise an exception and return -1 on error. */
int _PyTime_GetPerfCounterWithInfo(_PyTime_t* t, _Py_clock_info_t* info);

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
