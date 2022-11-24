/*
 * MessagePack for Python unpacking routine
 *
 * Copyright (C) 2009 Naoki INADA
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
#pragma once

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/runtime_value.h>

// clang-format off

#define MSGPACK_EMBED_STACK_SIZE  (1024)
#include "unpack_define.h"

namespace matxscript {
namespace runtime {
namespace serialization {

typedef MATXScriptAny msgpack_unpack_object;

typedef int (*MessagePackExtensionCallBack)(int8_t typecode,
                                            const char* pos,
                                            unsigned int length,
                                            msgpack_unpack_object* o);

typedef struct unpack_user {
    bool use_list;
    MessagePackExtensionCallBack ext_hook;
} unpack_user;

struct unpack_context;
typedef struct unpack_context unpack_context;
typedef int (*execute_fn)(unpack_context *ctx, const char* data, size_t len, size_t* off);

static inline msgpack_unpack_object msgpack_unpack_object_init()
{
    return msgpack_unpack_object{0, 0, TypeIndex::kRuntimeNullptr};
}

static inline void msgpack_unpack_object_destroy(msgpack_unpack_object*o)
{
    RTValue::DestroyCHost(o);
}

static inline msgpack_unpack_object unpack_callback_root(unpack_user* u)
{
    return msgpack_unpack_object{0, 0, TypeIndex::kRuntimeNullptr};
}

static inline int unpack_callback_uint16(unpack_user* u, uint16_t d, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(d);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}
static inline int unpack_callback_uint8(unpack_user* u, uint8_t d, msgpack_unpack_object* o)
{
    return unpack_callback_uint16(u, d, o);
}


static inline int unpack_callback_uint32(unpack_user* u, uint32_t d, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(d);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_uint64(unpack_user* u, uint64_t d, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(d);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_int32(unpack_user* u, int32_t d, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(d);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_int16(unpack_user* u, int16_t d, msgpack_unpack_object* o)
{
    return unpack_callback_int32(u, d, o);
}

static inline int unpack_callback_int8(unpack_user* u, int8_t d, msgpack_unpack_object* o)
{
    return unpack_callback_int32(u, d, o);
}

static inline int unpack_callback_int64(unpack_user* u, int64_t d, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(d);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_double(unpack_user* u, double d, msgpack_unpack_object* o)
{
    o->data.v_float64 = d;
    o->code = TypeIndex::kRuntimeFloat;
    return 0;
}

static inline int unpack_callback_float(unpack_user* u, float d, msgpack_unpack_object* o)
{
    return unpack_callback_double(u, d, o);
}

static inline int unpack_callback_nil(unpack_user* u, msgpack_unpack_object* o)
{
    o->code = TypeIndex::kRuntimeNullptr;
    return 0;
}

static inline int unpack_callback_true(unpack_user* u, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(1);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_false(unpack_user* u, msgpack_unpack_object* o)
{
    o->data.v_int64 = int64_t(0);
    o->code = TypeIndex::kRuntimeInteger;
    return 0;
}

static inline int unpack_callback_array(unpack_user* u, unsigned int n, msgpack_unpack_object* o)
{
    if (u->use_list) {
        List li;
        li.reserve(n);
        RTValue(std::move(li)).MoveToCHost(o);
    } else {
        auto new_node = make_inplace_array_object<TupleNode, RTValue>(n);
        new_node->size = 0;
        RTValue(Tuple(std::move(new_node))).MoveToCHost(o);
    }
    return 0;
}

static inline int unpack_callback_array_item(unpack_user* u, unsigned int current, msgpack_unpack_object* c, msgpack_unpack_object o)
{
    if (u->use_list) {
        reinterpret_cast<ListNode*>(c->data.v_handle)->emplace_back(RTValue::MoveFromCHost(&o));
    } else {
        auto* tup_node = reinterpret_cast<TupleNode*>(c->data.v_handle);
        tup_node->EmplaceInit(current, RTValue::MoveFromCHost(&o));
        // Only increment size after the initialization succeeds
        tup_node->size++;
    }
    return 0;
}

static inline int unpack_callback_array_end(unpack_user* u, msgpack_unpack_object* c)
{
    return 0;
}

static inline int unpack_callback_map(unpack_user* u, unsigned int n, msgpack_unpack_object* o)
{
    Dict d;
    d.reserve(n);
    RTValue(std::move(d)).MoveToCHost(o);
    return 0;
}

static inline int unpack_callback_map_item(unpack_user* u, unsigned int current, msgpack_unpack_object* c, msgpack_unpack_object k, msgpack_unpack_object v)
{
    auto* dict_node = reinterpret_cast<DictNode*>(c->data.v_handle);
    dict_node->emplace(RTValue::MoveFromCHost(&k), RTValue::MoveFromCHost(&v));
    return 0;
}

static inline int unpack_callback_map_end(unpack_user* u, msgpack_unpack_object* c)
{
    return 0;
}

static inline int unpack_callback_raw(unpack_user* u, const char* b, const char* p, unsigned int l, msgpack_unpack_object* o)
{
    RTValue(UTF8Decode(p, l)).MoveToCHost(o);
    return 0;
}

static inline int unpack_callback_bin(unpack_user* u, const char* b, const char* p, unsigned int l, msgpack_unpack_object* o)
{
    RTValue(String(p, l)).MoveToCHost(o);
    return 0;
}

typedef struct msgpack_timestamp {
    int64_t tv_sec;
    uint32_t tv_nsec;
} msgpack_timestamp;

/*
 * Unpack ext buffer to a timestamp. Pulled from msgpack-c timestamp.h.
 */
static int unpack_timestamp(const char* buf, unsigned int buflen, msgpack_timestamp* ts) {
    switch (buflen) {
    case 4:
        ts->tv_nsec = 0;
        {
            uint32_t v = _msgpack_load32(uint32_t, buf);
            ts->tv_sec = (int64_t)v;
        }
        return 0;
    case 8: {
        uint64_t value =_msgpack_load64(uint64_t, buf);
        ts->tv_nsec = (uint32_t)(value >> 34);
        ts->tv_sec = value & 0x00000003ffffffffLL;
        return 0;
    }
    case 12:
        ts->tv_nsec = _msgpack_load32(uint32_t, buf);
        ts->tv_sec = _msgpack_load64(int64_t, buf + 4);
        return 0;
    default:
        return -1;
    }
}

static int unpack_callback_ext(unpack_user* u, const char* base, const char* pos,
                               unsigned int length, msgpack_unpack_object* o)
{
    int8_t typecode = (int8_t)*pos++;
    // length also includes the typecode, so the actual data is length-1
    if (typecode == -1) {
        msgpack_timestamp ts;
        if (unpack_timestamp(pos, length-1, &ts) < 0) {
            return -1;
        }
        o->data.v_float64 = double(ts.tv_sec) + ts.tv_nsec/1000000000.0;
    } else {
        return u->ext_hook(typecode, pos, length-1, o);
    }
    return 0;
}

// clang-format on

}  // namespace serialization
}  // namespace runtime
}  // namespace matxscript

#include "unpack_template.h"
