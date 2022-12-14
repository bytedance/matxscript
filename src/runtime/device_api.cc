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

#include <matxscript/runtime/device_api.h>

#include <array>
#include <mutex>

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 32;
  // Get API
  static DeviceAPI* Get(const MATXScriptDevice& ctx) {
    return Get(ctx.device_type);
  }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }
  static void SetErrorMessage(int dev_type, String msg) {
    return Global()->SetAPIErrorMessage(dev_type, std::move(msg));
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  std::array<String, kMaxDeviceAPI> api_load_msg_;
  DeviceAPI* rpc_api_{nullptr};
  String rpc_load_msg_;
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
  }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager* inst = new DeviceAPIManager();
    return inst;
  }
  void SetAPIErrorMessage(int type, String msg) {
    if (type < kRPCSessMask) {
      std::lock_guard<std::mutex> lock(mutex_);
      api_load_msg_[type] = std::move(msg);
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      rpc_load_msg_ = std::move(msg);
    }
  }

  DeviceAPI* GetAPIImpl(int type, bool allow_missing) {
    String factory("device_api.");
    auto dev_name = DeviceTypeToName(type);
    factory.append(dev_name);
    auto* f = FunctionRegistry::Get(factory);
    if (f == nullptr) {
      if (allow_missing) {
        return nullptr;
      }
      if (type < api_load_msg_.size() && !api_load_msg_[type].empty()) {
        MXTHROW << api_load_msg_[kDLCUDA];
      } else {
        MXTHROW << dev_name << " device load failed!!!"
                << " maybe you need to fill env 'LD_LIBRARY_PATH'";
      }
      return nullptr;
    }
    void* ptr = (*f)({}).As<void*>();
    return static_cast<DeviceAPI*>(ptr);
  }

  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < api_.size()) {
      if (api_[type] != nullptr)
        return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr)
        return api_[type];
      api_[type] = GetAPIImpl(type, allow_missing);
      return api_[type];
    } else {
      if (!allow_missing) {
        MXTHROW << "device type " << type << " is not supported now!!!";
      }
      return nullptr;
    }
  }
};

DeviceAPI* DeviceAPI::Get(MATXScriptDevice device, bool allow_missing) {
  return DeviceAPIManager::Get(static_cast<int>(device.device_type), allow_missing);
}

void DeviceAPI::SetErrorMessage(MATXScriptDevice device, String msg) {
  return DeviceAPIManager::SetErrorMessage(static_cast<int>(device.device_type), std::move(msg));
}

MATXScriptStreamHandle DeviceAPI::CreateStream(MATXScriptDevice device) {
  MXTHROW << "Device does not support stream api.";
  return nullptr;
}

void DeviceAPI::FreeStream(MATXScriptDevice device, MATXScriptStreamHandle stream) {
  MXTHROW << "Device does not support stream api.";
}

void DeviceAPI::SyncStreamFromTo(MATXScriptDevice device,
                                 MATXScriptStreamHandle event_src,
                                 MATXScriptStreamHandle event_dst) {
  MXTHROW << "Device does not support stream api.";
}

DeviceStreamGuard::DeviceStreamGuard(MATXScriptDevice device, std::shared_ptr<void> stream) {
  this->device_ = device;
  this->new_stream_ = std::move(stream);
  this->device_api_ = DeviceAPI::Get(device_);
  this->old_stream_ = this->device_api_->GetSharedCurrentThreadStream(device_);
  this->device_api_->SetCurrentThreadStream(this->device_, this->new_stream_);
}

DeviceStreamGuard::~DeviceStreamGuard() {
  this->device_api_->SetCurrentThreadStream(this->device_, this->old_stream_);
}

const char* DeviceTypeToName(int type) {
  switch (type) {
    case kDLCPU:
      return "cpu";
    case kDLCUDA:
      return "cuda";
    case kDLCUDAHost:
      return "cuda_host";
    default:
      return "Unknown";
  }
}

int DeviceNameToType(const string_view& name) {
  static ska::flat_hash_map<string_view, DLDeviceType> name2type = {
      {"cpu", DLDeviceType::kDLCPU},
      {"gpu", DLDeviceType::kDLCUDA},
      {"cuda", DLDeviceType::kDLCUDA},
      {"cuda_host", DLDeviceType::kDLCUDAHost},
  };
  auto iter = name2type.find(name);
  if (iter == name2type.end()) {
    MXTHROW << "unsupported device name:" << name;
  }
  return iter->second;
}

std::ostream& operator<<(std::ostream& os, DLDevice dev) {  // NOLINT(*)
  int device_type = static_cast<int>(dev.device_type);
  if (device_type > kRPCSessMask) {
    os << "remote[" << (device_type / kRPCSessMask) << "]-";
    device_type = device_type % kRPCSessMask;
  }
  os << runtime::DeviceTypeToName(device_type) << "(" << dev.device_id << ")";
  return os;
}

}  // namespace runtime
}  // namespace matxscript
