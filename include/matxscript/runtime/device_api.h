// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the DeviceAPI originates from TVM.
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

#include <ostream>

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {
/*!
 * \brief the query type into GetAttr
 */
enum DeviceAttrKind : int {
  kExist = 0,
  kMaxThreadsPerBlock = 1,
  kWarpSize = 2,
  kMaxSharedMemoryPerBlock = 3,
  kComputeVersion = 4,
  kDeviceName = 5,
  kMaxClockRate = 6,
  kMultiProcessorCount = 7,
  kMaxThreadDimensions = 8,
  kMaxRegistersPerBlock = 9,
  kGcnArch = 10,
  kApiVersion = 11
};

/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 128;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 128;

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*!
 *  \brief MATX Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class MATX_DLL DeviceAPI {
 public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {
  }
  /*!
   * \brief Set the environment device id to ctx
   * \param ctx The context to be set.
   */
  virtual void SetDevice(MATXScriptDevice ctx) = 0;
  /*!
   * \brief Get attribute of specified device.
   * \param ctx The device context
   * \param kind The result kind
   * \param rv The return value.
   * \sa DeviceAttrKind
   */
  virtual void GetAttr(MATXScriptDevice ctx, DeviceAttrKind kind, RTValue* rv) = 0;
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param nbytes The number of bytes in memory.
   * \param alignment The alignment of the memory.
   * \param type_hint The type of elements. Only needed by certain backends such
   * as OpenGL, as nbytes & alignment are sufficient for most backends.
   * \return The allocated device pointer.
   */

  virtual void* AllocRaw(MATXScriptDevice ctx,
                         size_t nbytes,
                         size_t alignment,
                         DLDataType type_hint) = 0;

  virtual void* Alloc(MATXScriptDevice ctx,
                      size_t nbytes,
                      size_t alignment,
                      DLDataType type_hint) = 0;

  virtual void* Alloc(MATXScriptDevice ctx, size_t nbytes) = 0;

  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeRaw(MATXScriptDevice ctx, void* ptr) = 0;
  virtual void Free(MATXScriptDevice ctx, void* ptr) = 0;

  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param num_bytes The size of the memory in bytes
   * \param ctx_from The source context
   * \param ctx_to The target context
   * \param type_hint The type of elements, only neded by certain backends.
   *                  can be useful for cross device endian converison.
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void* from,
                              size_t from_offset,
                              void* to,
                              size_t to_offset,
                              size_t num_bytes,
                              MATXScriptDevice ctx_from,
                              MATXScriptDevice ctx_to,
                              DLDataType type_hint,
                              MATXScriptStreamHandle stream) = 0;
  /*!
   * \brief Create a new stream of execution.
   *
   * \param ctx The context of allocation.
   */
  virtual MATXScriptStreamHandle CreateStream(MATXScriptDevice ctx) = 0;

  /*!
   * \brief Free a stream of execution
   *
   * \param ctx The context of the stream
   * \param stream The pointer to be freed.
   */
  virtual void FreeStream(MATXScriptDevice ctx, MATXScriptStreamHandle stream) = 0;

  /*!
   * \brief return default stream on ctx
   *
   * \param ctx The context to perform operation.
   */
  virtual MATXScriptStreamHandle GetDefaultStream(MATXScriptDevice ctx) = 0;

  /*!
   * \brief return current stream on ctx
   *
   * \param ctx The context to perform operation.
   */
  virtual MATXScriptStreamHandle GetCurrentThreadStream(MATXScriptDevice ctx) = 0;
  virtual std::shared_ptr<void> GetSharedCurrentThreadStream(MATXScriptDevice ctx) = 0;

  /*!
   * \brief Set the stream only for current thread
   * \param ctx The context to set stream.
   * \param stream The stream to be set.
   */
  virtual void SetCurrentThreadStream(MATXScriptDevice ctx, std::shared_ptr<void> stream) = 0;

  /*!
   * \brief Synchronize the stream
   * \param ctx The context to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(MATXScriptDevice ctx, MATXScriptStreamHandle stream) = 0;

  virtual void CreateEventSync(MATXScriptStreamHandle stream) = 0;

  inline void CurrentThreadStreamSync(MATXScriptDevice ctx) {
    this->CreateEventSync(this->GetCurrentThreadStream(ctx));
  }

  /*!
   * \brief Synchronize 2 streams of execution.
   *
   * An event is created in event_src stream that the second then
   * stream waits on.  Neither event_src or event_dst need to be of
   * the same device ID as the context, but they must be of the same
   * device type.
   *
   * \param ctx The context of the streams.
   * \param event_src The source stream to synchronize.
   * \param event_dst The destination stream to synchronize.
   */
  virtual void SyncStreamFromTo(MATXScriptDevice ctx,
                                MATXScriptStreamHandle event_src,
                                MATXScriptStreamHandle event_dst);

  /*!
   * \brief Get device API based on context.
   * \param ctx The context
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  static DeviceAPI* Get(MATXScriptDevice ctx, bool allow_missing = false);
  static void SetErrorMessage(MATXScriptDevice ctx, String msg);

  /*!
   * \brief Whether a certian device type requires set device context
   *        before launching the kernel function.
   * \param device_type The device type.
   */
  static bool NeedSetDeviceContext(int device_type) {
    return device_type != kDLCPU;
  }
};

class DeviceStreamGuard {
  DeviceStreamGuard(MATXScriptDevice ctx, std::shared_ptr<void> stream);
  virtual ~DeviceStreamGuard();

 private:
  MATXScriptDevice device_;
  DeviceAPI* device_api_;
  std::shared_ptr<void> old_stream_;
  std::shared_ptr<void> new_stream_;
};

/*! \brief The device type bigger than this is RPC device */
constexpr int kRPCSessMask = 128;

const char* DeviceTypeToName(int type);
int DeviceNameToType(const string_view& name);

std::ostream& operator<<(std::ostream& os, DLDevice dev);

}  // namespace runtime
}  // namespace matxscript
