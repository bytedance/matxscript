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
#include <matxscript/pipeline/attributes.h>
#include <matxscript/pipeline/global_unique_index.h>
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container/user_data_ref.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/threadpool/i_thread_pool.h>

namespace {

using namespace ::matxscript::runtime;

class MySimpleNativeDataExample {
 public:
  MySimpleNativeDataExample() : content("hello") {
  }
  ~MySimpleNativeDataExample() = default;

  String get_content() const {
    return content;
  }

 public:
  String content;
};

MATX_REGISTER_NATIVE_OBJECT(MySimpleNativeDataExample)
    .SetConstructor<MySimpleNativeDataExample()>()
    .def("get_content", [](void* self) -> String {
      return reinterpret_cast<MySimpleNativeDataExample*>(self)->get_content();
    });

class MyNativeDataExample : public OpKernel {
 public:
  void Init() override {
    location_ = GetAttr<Unicode>("location").encode();
    abs_path_ = resource_path_ + location_;
    MXCHECK(FileUtil::Exists(abs_path_)) << "location is not valid, location: " << abs_path_;
  }

  RTValue Process(PyArgs inputs) const override {
    return get_content();
  }

  int Bundle(string_view folder) override {
    auto new_loc = BundlePath(location_, folder);
    SetAttr("location", new_loc);
    return 0;
  }

 public:
  const std::string& get_path() const {
    return abs_path_;
  }

  List get_content() const {
    List result;
    FileReader reader(this->get_path());
    const char* line;
    size_t len = 0;
    while (reader.ReadLine(&line, &len)) {
      result.push_back(String(line, len).decode());
    }
    return result;
  }

 private:
  String location_;
  std::string abs_path_;
};

MATX_REGISTER_NATIVE_OBJECT(MyNativeDataExample)
    .SetConstructor([](Unicode location) -> std::shared_ptr<void> {
      Attributes attrs;
      attrs.SetAttr<Unicode>("location", std::move(location));
      auto op = std::make_shared<MyNativeDataExample>();
      op->Initialize(attrs);
      return op;
    })
    .def("get_content", [](void* self) -> List {
      return reinterpret_cast<MyNativeDataExample*>(self)->get_content();
    });

class MyDeviceOpExample {
 public:
  MyDeviceOpExample(PyArgs args) {
    auto view = args[0].AsObjectView<Dict>();
    const auto& info = view.data();
    device_id_ = info["device_id"].As<int64_t>();
    session_device_id_ = info["session_device_id"].As<int64_t>();
    pool_ = static_cast<internal::IThreadPool*>(info["thread_pool"].As<void*>());
  }
  RTValue device_check(PyArgs args) {
    String prefix = args[0].As<String>();
    String output;
    output = prefix + ":" + std::to_string(device_id_).c_str() + ":" +
             std::to_string(session_device_id_);
    return output;
  }
  RTValue pool_size(PyArgs args) {
    if (pool_ == nullptr) {
      return 0;
    } else {
      return uint64_t(pool_->GetThreadsNum());
    }
  }

 private:
  int device_id_;
  int session_device_id_;
  internal::IThreadPool* pool_;
};

MATX_REGISTER_NATIVE_OBJECT(MyDeviceOpExample)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<MyDeviceOpExample>(args);
    })
    .RegisterFunction("device_check",
                      [](void* self, PyArgs args) -> RTValue {
                        return reinterpret_cast<MyDeviceOpExample*>(self)->device_check(args);
                      })
    .RegisterFunction("pool_size", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<MyDeviceOpExample*>(self)->pool_size(args);
    });

class EchoServiceExample {
 public:
  EchoServiceExample() {
  }
  virtual ~EchoServiceExample() = default;

  int echo(const MySimpleNativeDataExample& req, MySimpleNativeDataExample& rsp) {
    std::cout << "req content: " << req.get_content() << std::endl;
    rsp.content = "[Response] " + req.content;
    return 0;
  }
};

MATX_REGISTER_NATIVE_OBJECT(EchoServiceExample)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<EchoServiceExample>();
    })
    .RegisterFunction("echo", [](void* self, PyArgs args) -> RTValue {
      auto p = reinterpret_cast<EchoServiceExample*>(self);

      auto ud0 = args[0].AsObjectView<UserDataRef>();
      auto udp0 = (NativeObject*)(ud0.data()->ud_ptr);
      auto req = static_cast<MySimpleNativeDataExample*>(udp0->opaque_ptr_.get());

      auto ud1 = args[1].AsObjectView<UserDataRef>();
      auto udp1 = (NativeObject*)(ud1.data()->ud_ptr);
      auto rsp = static_cast<MySimpleNativeDataExample*>(udp1->opaque_ptr_.get());
      return p->echo(*req, *rsp);
    });

}  // namespace
