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
#include <matxscript/runtime/msgpack/msgpack.h>

#include <stddef.h>
#include <stdint.h>
#include <memory>

#include "pack.h"
#include "unpack.h"

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {
namespace serialization {

/******************************************************************************
 * pack
 *****************************************************************************/

static constexpr long long ITEM_LIMIT = (2LL << 31) - 1;

struct MessagePackerOptions {
  bool use_single_float = false;
  bool autoreset = true;
  bool use_bin_type = true;
  bool strict_types = false;
  bool datetime = false;
  size_t buf_size = 1024 * 1024;
};

struct BasicMessagePacker {
 public:
  explicit BasicMessagePacker(MessagePackerOptions options);
  virtual ~BasicMessagePacker();

  BasicMessagePacker(BasicMessagePacker&&) noexcept = default;
  BasicMessagePacker& operator=(BasicMessagePacker&&) noexcept = default;
  BasicMessagePacker(const BasicMessagePacker&) = delete;
  BasicMessagePacker& operator=(const BasicMessagePacker&) = delete;

 public:
  int pack(const Any& o, int nest_limit);

 private:
  MessagePackerOptions options_;
  msgpack_packer msg_pk_;

  friend class MessagePacker;
};

BasicMessagePacker::BasicMessagePacker(MessagePackerOptions options) {
  this->options_ = options;
  this->msg_pk_.buf = (char*)malloc(this->options_.buf_size);
  if (this->msg_pk_.buf) {
    this->msg_pk_.buf_size = this->options_.buf_size;
  } else {
    this->msg_pk_.buf_size = 0;
  }
  this->msg_pk_.length = 0;
  this->msg_pk_.use_bin_type = this->options_.use_bin_type;
}

BasicMessagePacker::~BasicMessagePacker() {
  if (this->msg_pk_.buf) {
    free(this->msg_pk_.buf);
  }
}

#define CHECK_MSGPACK_CALL(ret, ...)  \
  if ((ret) != 0) {                   \
    THROW_PY_ValueError(__VA_ARGS__); \
  }

int BasicMessagePacker::pack(const Any& o, int nest_limit) {
  int ret;
  size_t L;

  if (nest_limit < 0) {
    THROW_PY_ValueError("recursion limit exceeded.");
  }

  switch (o.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
      ret = msgpack_pack_nil(&this->msg_pk_);
    } break;
    case TypeIndex::kRuntimeInteger: {
      ret = msgpack_pack_long_long(&this->msg_pk_, o.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      if (this->options_.use_single_float) {
        ret = msgpack_pack_float(&this->msg_pk_, o.AsNoCheck<float>());
      } else {
        ret = msgpack_pack_double(&this->msg_pk_, o.AsNoCheck<double>());
      }
    } break;
    case TypeIndex::kRuntimeString: {
      auto s = o.AsNoCheck<string_view>();
      if (s.length() > ITEM_LIMIT) {
        THROW_PY_ValueError("bytes object is too large: ", s.substr(0, 200));
      }
      ret = msgpack_pack_bin(&this->msg_pk_, s.length());
      if (ret == 0) {
        ret = msgpack_pack_raw_body(&this->msg_pk_, s.data(), s.length());
      }
    } break;
    case TypeIndex::kRuntimeUnicode: {
      auto u = o.AsNoCheck<unicode_view>();
      auto bytes = UTF8Encode(u.data(), u.size());
      if (bytes.length() > ITEM_LIMIT) {
        THROW_PY_ValueError("str object is too large: ", bytes.substr(0, 200));
      }
      ret = msgpack_pack_raw(&this->msg_pk_, bytes.length());
      if (ret == 0) {
        ret = msgpack_pack_raw_body(&this->msg_pk_, bytes.data(), bytes.length());
      }
    } break;
    case TypeIndex::kRuntimeList: {
      auto v = o.AsObjectRefNoCheck<List>();
      L = v.size();
      if (L > ITEM_LIMIT) {
        THROW_PY_ValueError("list is too large");
      }
      ret = msgpack_pack_array(&this->msg_pk_, L);
      if (ret == 0) {
        auto iter = v.begin();
        auto iter_end = v.end();
        for (; iter != iter_end; ++iter) {
          ret = this->pack(*iter, nest_limit - 1);
          if (ret != 0) {
            break;
          }
        }
      }
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto v = o.AsObjectRefNoCheck<Tuple>();
      L = v.size();
      if (L > ITEM_LIMIT) {
        THROW_PY_ValueError("tuple is too large");
      }
      ret = msgpack_pack_array(&this->msg_pk_, L);
      if (ret == 0) {
        auto iter = v.begin();
        auto iter_end = v.end();
        for (; iter != iter_end; ++iter) {
          ret = this->pack(*iter, nest_limit - 1);
          if (ret != 0) {
            break;
          }
        }
      }
    } break;
    case TypeIndex::kRuntimeDict: {
      auto v = o.AsObjectRefNoCheck<Dict>();
      L = v.size();
      if (L > ITEM_LIMIT) {
        THROW_PY_ValueError("dict is too large");
      }
      ret = msgpack_pack_map(&this->msg_pk_, L);
      if (ret == 0) {
        auto items = v.items();
        for (const auto& kv : items) {
          ret = this->pack(kv.first, nest_limit - 1);
          if (ret != 0) {
            break;
          }
          ret = this->pack(kv.second, nest_limit - 1);
          if (ret != 0) {
            break;
          }
        }
      }
    } break;
    case TypeIndex::kRuntimeSet: {
      BasicMessagePacker set_packer(this->options_);
      auto v = o.AsObjectRefNoCheck<Set>();
      L = v.size();
      if (L > ITEM_LIMIT) {
        THROW_PY_ValueError("set is too large");
      }
      ret = msgpack_pack_array(&set_packer.msg_pk_, L);
      if (ret == 0) {
        auto iter = v.begin();
        auto iter_end = v.end();
        for (; iter != iter_end; ++iter) {
          ret = set_packer.pack(*iter, nest_limit - 1);
          if (ret != 0) {
            break;
          }
        }
      }
      CHECK_MSGPACK_CALL(ret, "pack set failed");
      ret = msgpack_pack_ext(&this->msg_pk_, TypeIndex::kRuntimeSet, set_packer.msg_pk_.length);
      CHECK_MSGPACK_CALL(ret, "pack set failed");
      ret =
          msgpack_pack_raw_body(&this->msg_pk_, set_packer.msg_pk_.buf, set_packer.msg_pk_.length);
      CHECK_MSGPACK_CALL(ret, "pack set failed");
    } break;
    case TypeIndex::kRuntimeNDArray: {
      auto v = o.AsObjectRefNoCheck<NDArray>();
      const DLTensor* dl_tensor = v.operator->();
      auto data_size = GetDataSize(*dl_tensor);
      L = sizeof(DLDataType)                                       // dtype
          + sizeof(int32_t) + (sizeof(int64_t) * dl_tensor->ndim)  // shape
          + data_size;                                             // data
      ret = msgpack_pack_ext(&this->msg_pk_, TypeIndex::kRuntimeNDArray, L);
      CHECK_MSGPACK_CALL(ret, "pack ndarray failed");
      ret = msgpack_pack_extend_write_buf(&this->msg_pk_, L);
      CHECK_MSGPACK_CALL(ret, "pack ndarray failed");
      char* write_buf = this->msg_pk_.buf + this->msg_pk_.length - L;
      // dtype
      std::memcpy(write_buf, &(dl_tensor->dtype.code), 1);
      write_buf++;
      std::memcpy(write_buf, &(dl_tensor->dtype.bits), 1);
      write_buf++;
      std::memcpy(write_buf, &(dl_tensor->dtype.lanes), 2);
      write_buf += 2;

      // shape
      int32_t ndim = dl_tensor->ndim;
      std::memcpy(write_buf, &ndim, sizeof(int32_t));
      write_buf += sizeof(int32_t);
      for (auto i = 0; i < dl_tensor->ndim; ++i) {
        int64_t shape = dl_tensor->shape[i];
        std::memcpy(write_buf, &shape, sizeof(int64_t));
        write_buf += sizeof(int64_t);
      }

      // data
      v.CopyToBytes(write_buf, data_size);
    } break;
    default: {
      THROW_PY_TypeError("can not serialize '", o.type_name(), "' object");
    } break;
  }
  return ret;
}

struct MessagePacker {
  static constexpr int DEFAULT_RECURSE_LIMIT = 511;

 public:
  static String pack(const Any& o,
                     const MessagePackerOptions& options,
                     int nest_limit = DEFAULT_RECURSE_LIMIT);
};

String MessagePacker::pack(const Any& o, const MessagePackerOptions& options, int nest_limit) {
  BasicMessagePacker packer(options);
  int ret = packer.pack(o, nest_limit);
  if (ret != 0) {
    THROW_PY_RuntimeError("msgpack: serialization error");
  }
  return String{packer.msg_pk_.buf, packer.msg_pk_.length};
}

/******************************************************************************
 * unpack
 *****************************************************************************/

struct MessageUnpacker {
  static constexpr int DEFAULT_RECURSE_LIMIT = 511;

 public:
  static int custom_ext_callback(int8_t typecode,
                                 const char* pos,
                                 unsigned int length,
                                 msgpack_unpack_object* o);

  static RTValue unpackb(const string_view& packed, bool use_list);
};

int MessageUnpacker::custom_ext_callback(int8_t typecode,
                                         const char* pos,
                                         unsigned int length,
                                         msgpack_unpack_object* o) {
  switch (typecode) {
    case TypeIndex::kRuntimeSet: {
      auto value = MessageUnpacker::unpackb(string_view(pos, length), false);
      if (!value.IsObjectRef<Tuple>()) {
        THROW_PY_ValueError("Unpack failed: Set Format Error");
      }
      auto tup = value.AsNoCheck<Tuple>();
      RTValue(Set(static_cast<const Any*>(tup.begin()), static_cast<const Any*>(tup.end())))
          .MoveToCHost(o);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      const char* buf = pos;
      MXCHECK(length >= (4 + sizeof(int32_t))) << "Msgpack: Invalid NDArray Data Format";
      // dtype
      DLDataType dtype;
      std::memcpy(&dtype.code, buf, 1);
      buf += 1;
      std::memcpy(&dtype.bits, buf, 1);
      buf += 1;
      std::memcpy(&dtype.lanes, buf, 2);
      buf += 2;
      // shape
      int32_t ndim;
      std::memcpy(&ndim, buf, sizeof(int32_t));
      buf += sizeof(int32_t);
      MXCHECK(length >= (4 + sizeof(int32_t) + ndim * sizeof(int64_t)))
          << "Msgpack: Invalid NDArray Data Format";
      std::vector<int64_t> shapes;
      shapes.reserve(ndim);
      size_t size = 1;
      for (int i = 0; i < ndim; ++i) {
        int64_t shape;
        std::memcpy(&shape, buf, sizeof(int64_t));
        buf += sizeof(int64_t);
        shapes.emplace_back(shape);
        size *= static_cast<size_t>(shape);
      }
      size *= (dtype.bits * dtype.lanes + 7) / 8;
      MXCHECK_EQ(length, (4 + sizeof(int32_t) + ndim * sizeof(int64_t) + size))
          << "Msgpack: Invalid NDArray Data Format";
      // data
      DLDevice device{kDLCPU, 0};
      auto arr = NDArray::Empty(std::move(shapes), dtype, device);
      arr.CopyFromBytes(buf, size);
      RTValue(std::move(arr)).MoveToCHost(o);
    } break;
    default: {
      THROW_PY_ValueError("Unpack failed: unknown ext type code: ", typecode);
      return -2;
    } break;
  }
  return 0;
}

RTValue MessageUnpacker::unpackb(const string_view& packed, bool use_list) {
  unpack_context ctx;
  size_t off = 0;
  int ret;
  const char* buf = packed.data();
  size_t buf_len = packed.size();

  unpack_init(&ctx);
  ctx.user.use_list = use_list;
  ctx.user.ext_hook = MessageUnpacker::custom_ext_callback;

  ret = unpack_construct(&ctx, buf, buf_len, &off);
  if (ret == 1) {
    auto obj = unpack_data(&ctx);
    if (off < buf_len) {
      THROW_PY_ValueError("ExtraData: ", string_view(buf + off, buf_len - off));
    }
    return RTValue::MoveFromCHost(&obj);
  }
  unpack_clear(&ctx);
  if (ret == 0) {
    THROW_PY_ValueError("Unpack failed: incomplete input");
  } else if (ret == -2) {
    THROW_PY_ValueError("Unpack failed: FormatError");
  } else if (ret == -3) {
    THROW_PY_ValueError("Unpack failed: StackError");
  }
  THROW_PY_ValueError("Unpack failed: error = ", ret);
  return None;
}

/******************************************************************************
 * export interface
 *****************************************************************************/

RTValue msgpack_loads(const string_view& s) {
  return MessageUnpacker::unpackb(s, true);
}

String msgpack_dumps(const Any& obj) {
  MessagePackerOptions options;
  return MessagePacker::pack(obj, options);
}

}  // namespace serialization
}  // namespace runtime
}  // namespace matxscript
