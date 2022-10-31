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
#include <matxscript/runtime/container/_ft_object_base.h>

#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/demangle.h>

namespace matxscript {
namespace runtime {

RTValue FTObjectBase::generic_call_attr(string_view func_name, PyArgs args) const {
  MX_CHECK_DPTR(FTObjectBase);
  auto func_iter = d->child_function_table_->find(func_name);
  if (func_iter == d->child_function_table_->end()) {
    MXTHROW << "class method not found " << DemangleType(d->child_type_index_->name())
            << "::" << func_name;
  }
  return func_iter->second(*this, args);
}

uint32_t FTObjectBaseNode::_RegisterRuntimeTypeIndex(string_view key,
                                                     uint32_t static_tindex,
                                                     uint32_t parent_tindex,
                                                     uint32_t num_child_slots,
                                                     bool child_slots_can_overflow) {
  return Object::GetOrAllocRuntimeTypeIndex(
      key, static_tindex, parent_tindex, num_child_slots, child_slots_can_overflow);
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(FTObjectBaseNode);

static MATXSCRIPT_ATTRIBUTE_UNUSED uint32_t __make_Object_tid_ft_list_0 =
    FTObjectBaseNode::_RegisterRuntimeTypeIndex("FTList",
                                                TypeIndex::kRuntimeFTList,
                                                FTObjectBaseNode::_GetOrAllocRuntimeTypeIndex(),
                                                0,
                                                true);
static MATXSCRIPT_ATTRIBUTE_UNUSED uint32_t __make_Object_tid_ft_dict_0 =
    FTObjectBaseNode::_RegisterRuntimeTypeIndex("FTDict",
                                                TypeIndex::kRuntimeFTDict,
                                                FTObjectBaseNode::_GetOrAllocRuntimeTypeIndex(),
                                                0,
                                                true);
static MATXSCRIPT_ATTRIBUTE_UNUSED uint32_t __make_Object_tid_ft_set_0 =
    FTObjectBaseNode::_RegisterRuntimeTypeIndex("FTSet",
                                                TypeIndex::kRuntimeFTSet,
                                                FTObjectBaseNode::_GetOrAllocRuntimeTypeIndex(),
                                                0,
                                                true);

std::ostream& operator<<(std::ostream& os, FTObjectBase const& n) {
  switch (n->type_index()) {
    case TypeIndex::kRuntimeFTList: {
      os << '[';
      Iterator iter = n.generic_call_attr("__iter__", {}).As<Iterator>();
      bool is_not_first = false;
      bool has_next = iter.HasNext();
      while (has_next) {
        if (is_not_first) {
          os << ", ";
        }
        is_not_first = true;
        RTValue val = iter.Next(&has_next);
        if (val.IsString()) {
          os << "b'" << val.AsNoCheck<string_view>() << "'";
        } else if (val.IsUnicode()) {
          os << "\'" << val.AsNoCheck<unicode_view>() << "\'";
        } else {
          os << val;
        }
      }
      os << ']';
    } break;
    case TypeIndex::kRuntimeFTSet: {
      os << '{';
      Iterator iter = n.generic_call_attr("__iter__", {}).As<Iterator>();
      bool is_not_first = false;
      bool has_next = iter.HasNext();
      while (has_next) {
        if (is_not_first) {
          os << ", ";
        }
        is_not_first = true;
        RTValue val = iter.Next(&has_next);
        if (val.IsString()) {
          os << "b'" << val.AsNoCheck<string_view>() << "'";
        } else if (val.IsUnicode()) {
          os << "\'" << val.AsNoCheck<unicode_view>() << "\'";
        } else {
          os << val;
        }
      }
      os << '}';
    } break;
    case TypeIndex::kRuntimeFTDict: {
      os << '{';
      Iterator iter = n.generic_call_attr("items", {}).As<Iterator>();
      bool is_not_first = false;
      bool has_next = iter.HasNext();
      while (has_next) {
        if (is_not_first) {
          os << ", ";
        }
        is_not_first = true;
        Tuple val = iter.Next(&has_next).As<Tuple>();
        if (val[0].IsString()) {
          os << "b'" << val[0].As<string_view>() << "': ";
        } else if (val[0].IsUnicode()) {
          os << "\'" << val[0].As<unicode_view>() << "\': ";
        } else {
          os << val[0];
          os << ": ";
        }
        if (val[1].IsString()) {
          os << "b'" << val[1].As<string_view>() << "'";
        } else if (val[1].IsUnicode()) {
          os << "\'" << val[1].As<unicode_view>() << "\'";
        } else {
          os << val[1];
        }
      }
      os << '}';
    } break;
    default: {
      os << "FTObjectBase(addr: " << n.get() << ")";
    } break;
  }
  return os;
}

}  // namespace runtime
}  // namespace matxscript
