# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Callable, Any
import matx
from matx.contrib.cc import create_shared
import os
import ctypes
import unittest


code = R'''
#include <future>
#include <thread>

#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/future_wrap.h>

using namespace ::matxscript::runtime;

namespace {
MATXSCRIPT_REGISTER_GLOBAL("test.get_future").set_body([](PyArgs args) -> RTValue {
  std::future<int> ft = std::async(std::launch::async, []{ return 8; });
  auto shared_ft = std::make_shared<std::future<int>>(std::move(ft));
  return Future::make_future_udref([shared_ft]() {
      if (shared_ft.get()->valid()) {
        return shared_ft.get()->get();
      } else {
        return -1;
      }
  });
});
}
'''


class _Lazy:

    def __init__(self, fu: Callable) -> None:
        self._fu: matx.NativeObject = matx.make_native_object("Future", fu)

    def __call__(self) -> Any:
        return self._fu()


def get_one() -> int:
    return 1


class AddFive:

    def __init__(self, op: Callable) -> None:
        self._op: Callable = op

    def __call__(self) -> int:
        return self._op() + 5


class TestFutureInterface(unittest.TestCase):

    def test_py_future_eval(self):
        new_get_one = matx.script(get_one)
        NewAddFive = matx.script(AddFive)
        Lazy = matx.script(_Lazy)
        lazy2 = Lazy(NewAddFive(new_get_one))
        assert lazy2() == 6

    def test_py_future_compiling(self):

        @matx.script
        def _lazy_add() -> int:
            lazy = _Lazy(AddFive(get_one))
            return lazy()

        assert _lazy_add() == 6

    def test_c_extension(self):
        output = 'libmy_future.so'
        with open('my_future.cc', 'w') as fw:
            fw.write(code)
        if os.path.exists(output):
            os.remove(output)
        create_shared(output, ['my_future.cc'], ['-std=c++14', '-shared'] + matx.get_cflags())
        ctypes.CDLL(output)

        get_ft = matx.get_global_func("test.get_future")
        ft = get_ft()

        @matx.script
        def _lazy_run(fu: Callable) -> int:
            return fu()

        assert _lazy_run(ft) == 8


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
