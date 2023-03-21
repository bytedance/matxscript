// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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

/*!
 * \file src/ir/op.cc
 * \brief Primitive operators and intrinsics.
 */
#include <matxscript/ir/hlo_expr.h>

#include <memory>

#include <matxscript/ir/_base/attr_registry.h>
#include <matxscript/ir/op_expr.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/object_internal.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

using OpRegistry = AttrRegistry<OpRegEntry, Op>;

// find operator by name
const Op& Op::Get(const StringRef& name) {
  const OpRegEntry* reg = OpRegistry::Global()->Get(name);
  MXCHECK(reg != nullptr) << "AttributeError: Operator " << name << " is not registered";
  return reg->op();
}

OpRegEntry::OpRegEntry(uint32_t reg_index) {
  ObjectPtr<OpNode> n = make_object<OpNode>();
  n->index_ = reg_index;
  op_ = Op(n);
}

OpRegEntry& OpRegEntry::RegisterOrGet(const StringRef& name) {
  return OpRegistry::Global()->RegisterOrGet(name);
}

// Get attribute map by key
const AttrRegistryMapContainerMap<Op>& Op::GetAttrMapContainer(const StringRef& attr_name) {
  return OpRegistry::Global()->GetAttrMap(attr_name);
}

// Check if a key is present in the registry.
bool Op::HasAttrMap(const StringRef& attr_name) {
  return OpRegistry::Global()->HasAttrMap(attr_name);
}

// Resets attr of the OpAttrMap.
void OpRegEntry::reset_attr(const StringRef& attr_name) {
  OpRegistry::Global()->ResetAttr(attr_name, op_);
}

void OpRegEntry::UpdateAttr(const StringRef& key, RTValue value, int plevel) {
  OpRegistry::Global()->UpdateAttr(key, op_, value, plevel);
}

// Frontend APIs
MATXSCRIPT_REGISTER_GLOBAL("ir.ListOpNames").set_body_typed([]() {
  return OpRegistry::Global()->ListAllNames();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.GetOp").set_body_typed([](StringRef name) -> Op {
  return Op::Get(name);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.OpGetAttr")
    .set_body_typed([](Op op, StringRef attr_name) -> RTValue {
      auto op_map = Op::GetAttrMap<RTValue>(attr_name);
      RTValue rv;
      if (op_map.count(op)) {
        rv = op_map[op];
      }
      return rv;
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.OpSetAttr")
    .set_body_typed([](Op op, StringRef attr_name, runtime::RTValue value, int plevel) {
      auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
      reg.set_attr(attr_name, value, plevel);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.OpResetAttr").set_body_typed([](Op op, StringRef attr_name) {
  auto& reg = OpRegistry::Global()->RegisterOrGet(op->name);
  reg.reset_attr(attr_name);
});

ObjectPtr<Object> CreateOp(const String& name) {
  // Hack use TVMRetValue as exchange
  auto op = Op::Get(StringRef(name));
  MXCHECK(op.defined()) << "Cannot find op \'" << name << '\'';
  return ObjectInternal::GetObjectPtr(op);
}

MATXSCRIPT_REGISTER_NODE_TYPE(OpNode).set_creator(CreateOp).set_repr_bytes(
    [](const Object* n) -> String {
      return static_cast<const OpNode*>(n)->name.operator String();
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const OpNode*>(ref.get());
      p->stream << "Op(" << node->name << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Op>("", [](Op op, ObjectPath p, IRDocsifier d) -> Doc {
      return Dialect(d, "Op")->Call({LiteralDoc::Str(op->name, p->Attr("name"))});
    });

MATXSCRIPT_REGISTER_GLOBAL("ir._call_builtin_op").set_body([](PyArgs args) -> RTValue {
  Type ret_type = args[0].As<Type>();
  StringRef op_name = args[1].As<StringRef>();
  Array<BaseExpr> call_args;
  for (int i = 2; i < args.size(); i += 1) {
    auto i_code = args[i].type_code();
    switch (i_code) {
      case TypeIndex::kRuntimeInteger: {
        call_args.push_back(IntImm(DataType::Int(64), args[i].As<int64_t>()));
      } break;
      case TypeIndex::kRuntimeFloat: {
        call_args.push_back(FloatImm(DataType::Float(64), args[i].As<double>()));
      } break;
      case TypeIndex::kRuntimeUnicode:
      case TypeIndex::kRuntimeString: {
        call_args.push_back(StringImm(args[i].As<StringRef>()));
      } break;
      default: {
        if (args[i].IsObjectRef<BaseExpr>()) {
          call_args.push_back(args[i].As<BaseExpr>());
        } else if (args[i].IsObjectRef<StringRef>()) {
          call_args.push_back(StringImm(args[i].As<StringRef>()));
        } else {
          MXCHECK(false) << "ir._call_builtin_op, args[" << i
                         << "] type error, only support int/float/str/BaseExpr";
        }
      } break;
    }
  }
  return Call(ret_type, Op::Get("ir." + op_name), call_args);
});

}  // namespace ir
}  // namespace matxscript
