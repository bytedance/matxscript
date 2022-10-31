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
from matx.contrib.cc import create_shared
import matx
import os
import ctypes
from typing import List
import unittest


code = R'''
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/native_object_registry.h>

using namespace ::matxscript::runtime;

class MyFoo {
 public:
  MyFoo(Unicode tag) : tag(std::move(tag)) {
  }
  ~MyFoo() = default;
  List split(const Unicode& input) const {
    return input.split();
  }
  Unicode tag;
};

namespace {
MATX_REGISTER_NATIVE_OBJECT(MyFoo)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[MyFoo] Expect 1 arguments but get " << args.size();
      return std::make_shared<MyFoo>(args[0].As<Unicode>());
    })
    .RegisterFunction("split", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 1) << "[MyFoo][func: split] Expect 1 arguments but get "
                                 << args.size();
      return reinterpret_cast<MyFoo*>(self)->split(args[0].As<Unicode>());
    });
}'''

output = 'libmyfoo.so'

with open('my_foo.cc', 'w') as fw:
    fw.write(code)

if os.path.exists(output):
    os.remove(output)


create_shared(output, ['my_foo.cc'], ['-std=c++14', '-shared'] + matx.get_cflags())

ctypes.CDLL(output)


class MyFooWrapper:

    def __init__(self) -> None:
        self.foo: matx.NativeObject = matx.make_native_object("MyFoo", "88")

    def __call__(self, ss: str) -> List[str]:
        return self.foo.split(ss)


class TestBuiltinImpl(unittest.TestCase):

    def test_split(self):
        myfoo = matx.script(MyFooWrapper)()
        assert len(myfoo("hello world")) == 2


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
