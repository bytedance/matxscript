// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the codegen is inspired by TVM.
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
 * \file codegen_c.cc
 */
#include "codegen_c.h"

#include <math.h>

#include <cctype>
#include <iomanip>

#include <matxscript/runtime/function_name_rules.h>
#include <matxscript/runtime/str_escape.h>
#include <matxscript/runtime/string_util.h>

namespace matxscript {
namespace codegen {

using namespace ir;

void CodeGenC::Init(bool output_ssa) {
  print_ssa_form_ = output_ssa;
}

void CodeGenC::InitAllState() {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
  current_func_rt_type_ = VoidType();
  current_py_func_name_ = "";
}

void CodeGenC::InitFuncState(const BaseFunc& f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
  current_func_rt_type_ = f->GetReturnType();
  if (f->IsClassMember()) {
    current_py_func_name_ = f->GetBoundName().operator String();
  } else {
    current_py_func_name_ = f->GetGlobalName().operator String();
  }
}

void CodeGenC::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  GetUniqueName("extern");
  GetUniqueName("void");
  GetUniqueName("int");
  GetUniqueName("float");
  GetUniqueName("double");
  GetUniqueName("char");
  GetUniqueName("unsigned");
  GetUniqueName("short");
  GetUniqueName("long");
  GetUniqueName("if");
  GetUniqueName("else");
  GetUniqueName("switch");
  GetUniqueName("case");
  GetUniqueName("default");
  GetUniqueName("for");
  GetUniqueName("do");
  GetUniqueName("while");
  GetUniqueName("goto");
  GetUniqueName("register");
  GetUniqueName("continue");
  GetUniqueName("break");
  GetUniqueName("typedef");
  GetUniqueName("struct");
  GetUniqueName("enum");
  GetUniqueName("union");
  GetUniqueName("return");
}

void CodeGenC::PrintLineVars(std::ostream& os,
                             const Array<BaseExpr>& params,
                             const Array<BaseExpr>& default_params,
                             bool alloc_var,
                             bool with_var_name,
                             bool with_var_type,
                             bool with_defaults,
                             bool no_alias,
                             bool use_move,
                             bool skip_first) {
  MXCHECK(params.size() >= default_params.size());
  size_t default_begin_pos = params.size() - default_params.size();
  size_t params_begin_pos = skip_first ? 1 : 0;
  for (size_t i = params_begin_pos; i < params.size(); ++i) {
    if (i != params_begin_pos) {
      os << ", ";
    }
    BaseExpr v = params[i];
    String vid;
    if (alloc_var) {
      vid = AllocVarID(v);
    } else {
      vid = GetVarID(v);
    }
    if (with_var_type) {
      if (const auto* v_node = v.as<PrimVarNode>()) {
        if (v_node->dtype.is_handle()) {
          auto it = alloc_storage_scope_.find(v_node);
          if (it != alloc_storage_scope_.end()) {
            PrintStorageScope(it->second, os);
          }
          PrintType(GetType(Downcast<PrimVar>(v)), os);
          // Register handle data type
          // TODO(tvm-team): consider simply keep type info in the
          // type annotation(via a normalizing rewriting).
          if (auto* ptr = v_node->type_annotation.as<PointerTypeNode>()) {
            if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
              RegisterHandleType(v_node, prim->dtype);
            }
          }
          if (no_alias && restrict_keyword_.length() != 0) {
            os << ' ' << restrict_keyword_;
          }
        } else {
          PrintType(GetType(Downcast<PrimVar>(v)), os);
        }
      } else if (const auto* v_node = v.as<HLOVarNode>()) {
        PrintType(v_node->type_annotation, os);
      }
    }
    if (with_var_name) {
      if (use_move) {
        os << " std::move(" << vid << ")";
      } else {
        os << ' ' << vid;
      }
    }
    if (with_defaults) {
      MXCHECK(with_var_type && with_var_name);
      if (i >= default_begin_pos) {
        std::ostringstream temp_ss;
        VisitExpr(default_params[i - default_begin_pos], temp_ss);
        os << '=' << temp_ss.str();
      }
    }
  }
}

void CodeGenC::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  bool no_alias = f->HasNonzeroAttr(attr::kNoAlias);

  this->PrintFuncPrefix(f->ret_type);
  this->stream << " " << f->GetGlobalName() << "(";
  PrintLineVars(this->stream,
                Downcast<Array<BaseExpr>>(f->params),
                Downcast<Array<BaseExpr>>(f->default_params),
                true,
                true,
                true,
                false,
                no_alias);
  stream << ") {";
  this->PrintSpanWithNewLine(f->span, this->stream);

  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body, this->stream);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n\n";
}

void CodeGenC::AddFunctionDeclaration(const ir::BaseFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  String func_name = f->GetGlobalName();

  this->PrintFuncPrefix(f->GetReturnType());
  this->stream << " " << func_name << "(";
  PrintLineVars(this->stream, f->GetParams(), f->GetDefaultParams(), true, true, true, true);
  stream << ");\n";
  this->stream << "int " << FunctionNameRules::add_packed_suffix(func_name)
               << "(MATXScriptAny*, int, MATXScriptAny*, void*);\n";
}

void CodeGenC::AddFunction(const Function& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  bool no_alias = f->HasNonzeroAttr(attr::kNoAlias);
  if (f->IsClassMember()) {
    this->PrintType(f->ret_type, stream);
    this->stream << " " << f->GetBelongToClassName() << "::" << f->GetBoundName() << "(";
    PrintLineVars(
        this->stream, f->params, f->default_params, true, true, true, false, no_alias, false, true);
    // alloc implicit self var
    AllocVarID(f->params[0]);
    // alloc implicit session pointer var
    const auto& sess_var = GetImplicitClassSessionVar();
    AllocVarID(sess_var);
  } else {
    this->PrintFuncPrefix(f->ret_type);
    this->stream << " " << f->GetGlobalName() << "(";
    PrintLineVars(this->stream, f->params, f->default_params, true, true, true, false, no_alias);
  }

  stream << ") {";
  this->PrintSpanWithNewLine(f->span, this->stream);
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body, stream);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent(stream);
  this->stream << "}\n\n";
}

void CodeGenC::PrintFuncPrefix(ir::Type ret_type) {
  PrintType(ret_type, stream);
}

void CodeGenC::PrintFinalReturn() {
  current_func_rt_type_ = ir::VoidType();
  current_py_func_name_ = "";
}

void CodeGenC::PrintPackedFunctionMacro(const String& global_symbol,
                                        const String& bound_symbol,
                                        const Type& ret_type,
                                        const Array<BaseExpr>& args,
                                        const Array<BaseExpr>& default_args,
                                        bool first_arg_is_self,
                                        bool capture_session_handle,
                                        const Span& span) {
  auto py_info = this->GenPythonStyleSpanMessage(span, bound_symbol);
  std::function<Type(const Type& t)> remove_ref_type;
  remove_ref_type = [&remove_ref_type](const Type& t) -> Type {
    if (auto n = t.as<RefTypeNode>()) {
      return remove_ref_type(n->value);
    }
    return t;
  };
  auto ret_ty = remove_ref_type(ret_type);

  auto num_args = args.size();
  auto num_default_args = default_args.size();
  if (capture_session_handle) {
    --num_args;
    --num_default_args;
  }
  auto echo_fn_call = [&](int dynamic_num_args, const String& args_repr, bool fill_default) {
    ObjectType any_view_type(true, span);
    UserDataType user_data_type(span);
    this->PrintIndent(stream);
    if (IsVoidType(ret_ty)) {
      stream << global_symbol << "(";
    } else {
      stream << "auto ret = " << global_symbol << "(";
    }
    size_t i = 0;
    for (; i < dynamic_num_args; ++i) {
      if (i > 0) {
        stream << ", ";
      }
      auto& arg_i_ty = RemoveReference(args[i]->checked_type_);
      String args_value_i = args_repr + "[" + std::to_string(i) + "]";
      String args_repr_i;
      if (auto* node = args[i].as<PrimVarNode>()) {
        args_repr_i.append(node->name_hint.view());
      } else if (auto* node = args[i].as<HLOVarNode>()) {
        args_repr_i.append(node->name_hint().view());
      } else {
        static const char* arg_repr_eng[] = {
            "1st",
            "2nd",
            "3rd",
        };
        if (i < 3) {
          args_repr_i.append(arg_repr_eng[i]);
        } else {
          args_repr_i.append("the ");
          auto i_s = std::to_string(i);
          args_repr_i.append(args_repr_i.data(), args_repr_i.size());
          args_repr_i.append("th");
          args_repr_i.append(" argument");
        }
      }
      stream << PrintTypeCast(any_view_type, arg_i_ty, args_value_i, args_value_i, py_info);
    }
    if (fill_default) {
      int64_t default_begin_pos = num_args - num_default_args;
      auto fill_num = num_args - dynamic_num_args;
      for (; i < num_args; ++i) {
        if (i > 0) {
          stream << ", ";
        }
        this->VisitExpr(default_args[i - default_begin_pos], stream);
      }
    }
    if (capture_session_handle) {
      if (i > 0) {
        stream << ", ";
      }
      stream << "resource_handle";
    }
    stream << ");";
    this->PrintSpanWithNewLine(span, stream);
    if (IsVoidType(ret_ty)) {
      this->PrintIndent(stream);
      stream << "out_ret_value->code = TypeIndex::kRuntimeNullptr;\n";
    } else if (ret_ty.as<ClassTypeNode>()) {
      this->PrintIndent(stream);
      stream << "(ret.operator RTValue()).MoveToCHost(out_ret_value);\n";
    } else {
      this->PrintIndent(stream);
      stream << "RTValue(std::move(ret)).MoveToCHost(out_ret_value);\n";
    }
  };
  // declare c packed function
  stream << "int " << FunctionNameRules::add_packed_suffix(global_symbol) << "(";
  stream << "MATXScriptAny* args, int num_args, ";
  stream << "MATXScriptAny* out_ret_value, void* resource_handle = nullptr)\n";
  stream << "{\n";
  auto scope = this->BeginScope();

  // body
  this->PrintIndent(stream);
  stream << "TArgs args_t(args, num_args);\n\n";

  // check if has kwargs
  int kwargs_scope = 0;
  if (num_args > 0) {
    this->PrintIndent(stream);
    stream << "if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {\n";
    kwargs_scope = this->BeginScope();
    {
      // arg names
      this->PrintIndent(stream);
      stream << "string_view arg_names[" << num_args << "] {";
      for (auto i = 0; i < num_args; ++i) {
        if (i > 0) {
          stream << ", ";
        }
        stream << "\"";
        this->VisitExpr(args[i], stream);
        stream << "\"";
      }
      stream << "};\n";

      // default args
      if (num_default_args > 0) {
        this->PrintIndent(stream);
        stream << "static RTValue default_args[" << num_default_args << "]{";
        for (auto i = 0; i < num_default_args; ++i) {
          if (i > 0) {
            stream << ", ";
          }
          stream << "RTValue(";
          this->VisitExpr(default_args[i], stream);
          stream << ")";
        }
        stream << "};\n";
      }

      // UnpackHelper
      this->PrintIndent(stream);
      stream << "KwargsUnpackHelper helper(\"" << bound_symbol << "\", arg_names, " << num_args;
      if (num_default_args > 0) {
        stream << ", default_args, " << num_default_args;
      } else {
        stream << ", nullptr, " << num_default_args;
      }
      stream << ");\n";

      // pos_args
      this->PrintIndent(stream);
      stream << "RTView pos_args[" << num_args << "];\n";
      this->PrintIndent(stream);
      stream << "helper.unpack(pos_args, args, num_args);";
      this->PrintSpanWithNewLine(span, stream);
      echo_fn_call(num_args, "pos_args", false);
    }
    this->EndScope(kwargs_scope);
    this->PrintIndent(stream);
    stream << "} else {\n";
    kwargs_scope = this->BeginScope();
  }

  this->PrintIndent(stream);
  stream << "switch(num_args) {\n";
  auto arg_switch_scope = this->BeginScope();

  for (int32_t arg_i = num_args - num_default_args; arg_i <= num_args; ++arg_i) {
    this->PrintIndent(stream);
    stream << "case " << arg_i << ": {\n";
    auto arg_case_scope = this->BeginScope();
    echo_fn_call(arg_i, "args_t", true);
    this->EndScope(arg_case_scope);
    this->PrintIndent(stream);
    stream << "} break;\n";
  }

  this->PrintIndent(stream);
  stream << "default: {";
  stream << "THROW_PY_TypeError(" << py_info << ", \"" << bound_symbol << "() ";
  if (num_default_args == 0) {
    stream << "takes " << num_args;
  } else {
    stream << "takes from " << num_args - num_default_args << " to " << num_args;
  }
  stream << " positional arguments but \", num_args, \" were given\");";
  stream << "} break;";
  this->PrintSpanWithNewLine(span, stream);

  this->EndScope(arg_switch_scope);
  this->PrintIndent(stream);
  stream << "}\n";

  if (num_args > 0) {
    this->EndScope(kwargs_scope);
    this->PrintIndent(stream);
    stream << "}\n\n";
  }

  this->PrintIndent(stream);
  stream << "return 0;\n";

  this->EndScope(scope);
  this->PrintIndent(stream);
  stream << "}\n\n";
}

void CodeGenC::PrintPackedFunctionMacro(const ir::BaseFunc& f) {
  bool first_arg_is_self = false;
  if (!f->GetParams().empty()) {
    first_arg_is_self = f->GetParams()[0]->checked_type()->IsInstance<ClassTypeNode>();
  }
  String global_symbol = f->GetGlobalName().operator String();
  String bound_name;
  if (f->HasBoundName()) {
    bound_name = f->GetBoundName().operator String();
  } else {
    bound_name = global_symbol;
  }
  return PrintPackedFunctionMacro(global_symbol,
                                  bound_name,
                                  f->GetReturnType(),
                                  f->GetParams(),
                                  f->GetDefaultParams(),
                                  first_arg_is_self,
                                  f->CaptureSessionHandle(),
                                  f->span);
}

String CodeGenC::Finish() {
  return decl_stream.str() + stream.str();
}

void CodeGenC::PrintExpr(const BaseExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    if (const PrimExprNode* pe = n.as<PrimExprNode>()) {
      os << SSAGetID(temp.str(), pe->dtype, os);
    } else if (const HLOExprNode* hlo = n.as<HLOExprNode>()) {
      os << SSAGetID(temp.str(), hlo->checked_type_, os);
    } else {
      MXTHROW << "[BaseExpr:" << n->GetTypeKey() << "] is not supported";
    }
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenC::PrintSSAAssign(const String& target,
                              const String& src,
                              ir::Type t,
                              std::ostream& os) {
  PrintType(t, os);
  os << ' ' << target << " = ";
  if (src.length() > 3 && src[0] == '(' && src[src.length() - 1] == ')') {
    os << src.substr(1, src.length() - 2);
  } else {
    os << src;
  }
  os << ";\n";
}

// Print a reference expression to a buffer.
String CodeGenC::GetBufferRef(DataType t, const PrimVarNode* buffer, PrimExpr index) {
  std::ostringstream os;
  String vid = GetVarID(buffer);
  String scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = IsVolatile(buffer);
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t) || is_vol) {
      os << "((";
      if (is_vol) {
        os << "volatile ";
      }
      // Scope may not be part of type.
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      PrintType(t, os);
      os << "*)" << vid << ')';
    } else {
      os << vid;
    }
    os << "[(";
    PrintExpr(index, os);
    os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << ']';
  } else {
    // Buffer declared as vector type.
    // optimize for case where it is in register,
    if (HandleTypeMatch(buffer, t) && !is_vol) {
      // optimize for constant access
      if (auto* ptr = index.as<IntImmNode>()) {
        int64_t offset = ptr->value;
        MXCHECK_EQ(offset % t.lanes(), 0) << "Find unaligned vector load to a vector type";
        os << vid << '[' << (offset / t.lanes()) << ']';
        return os.str();
      }
    }
    os << "((";
    if (is_vol) {
      os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, os);
    }
    PrintType(t, os);
    os << "*)(";
    if (!HandleTypeMatch(buffer, t.element_of())) {
      os << '(';
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      PrintType(t.element_of(), os);
      os << "*)";
    }
    os << vid << " + (";
    PrintExpr(index, os);
    os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << "))[0]";
  }
  return os.str();
}

bool CodeGenC::HandleTypeMatch(const PrimVarNode* buf_var, DataType t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end())
    return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const PrimVarNode* buf_var, DataType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    MXCHECK(it->second == t) << "conflicting buf var type";
  }
}

String CodeGenC::CastFromTo(String value, DataType from, DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeGenC::PrintStorageSync(const PrimCallNode* op) {  // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const String& scope, std::ostream& os) {  // NOLINT(*)
  MXCHECK_EQ(scope, "global");
}

void CodeGenC::PrintSpan(const Span& span, std::ostream& os) {
  if (span.defined() && span->lineno >= 0 && !span->file_name.empty()) {
    os << "  // " << span->file_name << ":" << span->lineno;
  }
}

void CodeGenC::PrintSpanWithNewLine(const Span& span, std::ostream& os) {
  PrintSpan(span, os);
  os << "\n";
}

String CodeGenC::GenPythonStyleSpanMessage(const Span& span, const string_view& func) {
  if (span.defined() && span->lineno >= 0 && !span->file_name.empty()) {
    auto lineno = std::to_string(span->lineno);
    String message;
    message.append("\"");
    message.append("File \\\"").append(span->file_name).append("\\\"");
    message.append(", line ").append(lineno.data(), lineno.size());
    message.append(", in ").append(func).append("\\n");
    message.append("\"");
    return message;
  }
  return "\"\"";
}

void CodeGenC::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  MXCHECK_EQ(t.lanes(), 1) << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*";
    return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float";
      return;
    }
    if (t.bits() == 64) {
      os << "double";
      return;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "uint" << t.bits() << "_t";
        return;
      }
      case 1:
        os << "int";
        return;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "int" << t.bits() << "_t";
        return;
      }
    }
  }
  MXLOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenC::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (auto* ptr = type.as<ObjectTypeNode>()) {
    if (ptr->is_view) {
      os << "RTView";
    } else {
      os << "RTValue";
    }
  } else if (auto* ptr = type.as<UnicodeTypeNode>()) {
    if (ptr->is_view) {
      os << "unicode_view";
    } else {
      os << "Unicode";
    }
  } else if (auto* ptr = type.as<StringTypeNode>()) {
    if (ptr->is_view) {
      os << "string_view";
    } else {
      os << "String";
    }
  } else if (auto* ptr = type.as<ListTypeNode>()) {
    if (ptr->is_full_typed) {
      os << "FTList<";
      PrintType(ptr->item_type, os);
      os << ">";
    } else {
      os << "List";
    }
  } else if (auto* ptr = type.as<DictTypeNode>()) {
    if (ptr->is_full_typed) {
      os << "FTDict<";
      PrintType(ptr->key_type, os);
      os << ", ";
      PrintType(ptr->value_type, os);
      os << ">";
    } else {
      os << "Dict";
    }
  } else if (auto* ptr = type.as<SetTypeNode>()) {
    if (ptr->is_full_typed) {
      os << "FTSet<";
      PrintType(ptr->item_type, os);
      os << ">";
    } else {
      os << "Set";
    }
  } else if (auto* ptr = type.as<IteratorTypeNode>()) {
    if (ptr->container_type->IsInstance<ObjectTypeNode>()) {
      os << "Iterator";
    } else {
      PrintType(ptr->container_type, os);
      os << "::iterator";
    }
  } else if (auto* ptr = type.as<ExceptionTypeNode>()) {
    os << ptr->name;
  } else if (auto* ptr = type.as<FileTypeNode>()) {
    os << "File";
  } else if (IsVoidType(type)) {
    os << "void";
  } else if (auto* ptr = type.as<TupleTypeNode>()) {
    if (ptr->is_std_tuple) {
      os << "std::tuple<";
      for (auto ti = 0; ti < ptr->fields.size(); ++ti) {
        if (ti > 0) {
          os << ", ";
        }
        PrintType(ptr->fields[ti], os);
      }
      os << ">";
    } else {
      os << "Tuple";
    }
  } else if (auto* ptr = type.as<TrieTypeNode>()) {
    os << "Trie";
  } else if (auto* ptr = type.as<UserDataTypeNode>()) {
    os << "UserDataRef";
  } else if (auto* ptr = type.as<DynTensorTypeNode>()) {
    os << "NDArray";
  } else if (auto* ptr = type.as<RegexTypeNode>()) {
    os << "Regex";
  } else if (auto* ptr = type.as<OpaqueObjectTypeNode>()) {
    os << "OpaqueObject";
  } else if (auto* ptr = type.as<RefTypeNode>()) {
    os << "const ";
    PrintType(ptr->value, os);
    os << "&";
  } else if (auto* ptr = type.as<RangeTypeNode>()) {
    MXLOG(FATAL)
        << "RangeType should be decomposed by compiler. Please contact the developer to resolve the issue";
  } else {
    MXLOG(FATAL) << "Type " << type << " does not have a corresponding Runtime Type";
  }
}

String CodeGenC::PrintTypeCast(const Type& from_type,
                               const Type& to_type,
                               const string_view& value,
                               const string_view& value_repr,
                               const string_view& py_info) {
  const auto& from_type0 = RemoveReference(from_type);
  const auto& to_type0 = RemoveReference(to_type);

  std::ostringstream from_type_os;
  this->PrintType(from_type0, from_type_os);
  auto from_type_repr = from_type_os.str();

  std::ostringstream to_type_os;
  this->PrintType(to_type0, to_type_os);
  auto to_type_repr = to_type_os.str();

  if (IsBaseTypeOf(to_type0, from_type0, false)) {
    return String::Concat(
        {"CAST_TO_CLASS_VIEW_NOCHECK<", from_type_repr, ", ", to_type_repr, ">(", value, ")"});
  } else if (IsBaseTypeOf(from_type, to_type, false)) {
    return String::Concat(
        {"CAST_TO_CLASS_VIEW<", from_type_repr, ", ", to_type_repr, ">(", value, ")"});
  } else {
    if (to_type.as<ClassTypeNode>()) {
      // other to ClassType
      if (IsObjectType(from_type)) {
        // Any to ClassType
        return String::Concat({to_type_repr, "(static_cast<const Any&>(", value, "))"});
      } else if (IsUserDataType(from_type) || from_type.as<ClassTypeNode>()) {
        // UserData to ClassType
        return String::Concat({to_type_repr, "(", value, ")"});
      } else {
        MXTHROW << "Internal Error: can not convert '" << from_type->GetPythonTypeName() << "' to '"
                << to_type->GetPythonTypeName() << "'";
        return String{};
      }
    } else if (from_type.as<ClassTypeNode>()) {
      // ClassType to other
      if (IsUserDataType(to_type) || IsObjectType(to_type)) {
        return String::Concat({"(", value, ").operator ", to_type_repr, "()"});
      } else {
        MXTHROW << "Internal Error: can not convert '" << from_type->GetPythonTypeName() << "' to '"
                << to_type->GetPythonTypeName() << "'";
        return String{};
      }
    } else if (auto* from_node = from_type.as<ObjectTypeNode>()) {
      // Any to other
      auto* to_node = to_type.as<ObjectTypeNode>();
      if (to_node && to_node->is_view == from_node->is_view) {
        return String::Concat({"(", value, ")"});
      } else {
        string_view new_to_type_repr;
        if (IsUnicodeType(to_type0)) {
          new_to_type_repr = "py::str";
        } else if (IsStringType(to_type0)) {
          new_to_type_repr = "py::bytes";
        } else {
          new_to_type_repr = to_type_repr;
        }
        return String::Concat({"internal::TypeAsHelper<",
                               to_type_repr,
                               ">::run((",
                               value,
                               "), __FILE__, __LINE__, ",
                               py_info,
                               ", \"expect '",
                               BytesEscape(value_repr),
                               "' is '",
                               new_to_type_repr,
                               "' type\")"});
      }
    } else {
      if (IsObjectType(to_type)) {
        return String::Concat({to_type_repr, "(", value, ")"});
      } else {
        return String::Concat({"GenericValueConverter<", to_type_repr, ">{}(", value, ")"});
      }
    }
  }
  // should be unreachable
  MXTHROW << "Internal Error: can not convert '" << from_type->GetPythonTypeName() << "' to '"
          << to_type->GetPythonTypeName() << "'";
  return String{};
}

inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  if (op->dtype == DataType::Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->dtype, os);
    os << ")" << op->value;
  }
}

inline void PrintUIntConst(DataType dtype,
                           uint64_t val,
                           std::ostream& os,
                           CodeGenC* p) {  // NOLINT(*)
  if (dtype == DataType::UInt(32)) {
    std::ostringstream temp;
    temp << val << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(dtype, os);
    os << ")" << val;
  }
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->dtype.bits() == 32) {
          temp << "std::numeric_limits<float>::infinity()";
        } else {
          temp << "std::numeric_limits<double>::infinity()";
        }
      } else {
        if (op->dtype.bits() == 32) {
          temp << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
               << std::scientific << op->value << 'f';
        } else {
          temp << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
               << std::scientific << op->value;
        }
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->dtype, os);
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      MXLOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenC::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenC::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  auto bs = op->value.operator String();
  if (bs.size() > 0) {
    auto str_escape = runtime::BytesEscape(bs);
    os << "string_view(\"" << str_escape << "\", " << bs.size() << ")";
  } else {
    os << "string_view()";
  }
}
void CodeGenC::VisitExpr_(const UnicodeImmNode* op, std::ostream& os) {  // NOLINT(*)
  Unicode us = op->value.operator String().decode();
  if (us.size() > 0) {
    auto str_escape = runtime::UnicodeEscape(us);
    os << "unicode_view(U\"" << str_escape << "\", " << us.size() << ")";
  } else {
    os << "unicode_view()";
  }
}

template <typename T>
inline void PrintBinaryExpr(const T* op,
                            const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsic(const PrimCallNode* op,
                                 const char* opstr,
                                 std::ostream& os,  // NOLINT(*)
                                 CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    MXCHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->args[0], op->args[1], os);
  }
}
void CodeGenC::VisitExpr_(const PrimCastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
}
void CodeGenC::VisitExpr_(const HLOCastPrimNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value_os;
  this->PrintExpr(op->value, value_os);
  auto value_expr = value_os.str();
  auto& from_type = op->value->checked_type();

  PrimType to_type(op->dtype);

  if (from_type == to_type) {
    os << value_expr;
  } else {
    auto py_info = this->GenPythonStyleSpanMessage(op->span, this->current_py_func_name_);
    if (IsObjectType(RemoveReference(from_type))) {
      os << PrintTypeCast(from_type, to_type, value_expr, value_expr, py_info);
    } else {
      // TODO(mxd) : fix object cast
      os << "((";
      this->PrintType(op->dtype, os);
      os << ")" << value_expr << ")";
    }
  }
}
void CodeGenC::VisitExpr_(const PrimVarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenC::VisitExpr_(const PrimAddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenC::VisitExpr_(const PrimSubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenC::VisitExpr_(const PrimMulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenC::VisitExpr_(const PrimDivNode* op, std::ostream& os) {  // NOLINT(*)
  // PrintBinaryExpr(op, "/", os, this); // no check
  PrintBinaryExpr(op, "ArithOps::div", os, this);
}
void CodeGenC::VisitExpr_(const PrimFloorDivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "ArithOps::floordiv", os, this);
}
void CodeGenC::VisitExpr_(const PrimModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenC::VisitExpr_(const PrimFloorModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "ArithOps::floormod", os, this);
}
void CodeGenC::VisitExpr_(const PrimMinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenC::VisitExpr_(const PrimMaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenC::VisitExpr_(const PrimEQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenC::VisitExpr_(const PrimNENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenC::VisitExpr_(const PrimLTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenC::VisitExpr_(const PrimLENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenC::VisitExpr_(const PrimGTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenC::VisitExpr_(const PrimGENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenC::VisitExpr_(const PrimAndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenC::VisitExpr_(const PrimOrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenC::VisitExpr_(const PrimNotNode* op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenC::PrintCallExtern(Type ret_type,
                               StringRef global_symbol,
                               const Array<BaseExpr>& args,
                               bool skip_first_arg,
                               std::ostream& os) {  // NOLINT(*)
  os << global_symbol << "(";
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

void CodeGenC::VisitExpr_(const PrimCallNode* op, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);

    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      MXCHECK_GE(op->args.size(), 1U);
      auto func = Downcast<StringImm>(op->args[0]);
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)),
                            func->value,
                            Downcast<Array<BaseExpr>>(op->args),
                            true,
                            os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)),
                            op_attr_global_symbol_[call_op],
                            Downcast<Array<BaseExpr>>(op->args),
                            false,
                            os);
    } else if (op->op.same_as(builtin::bitwise_and())) {
      PrintBinaryIntrinsic(op, " & ", os, this);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      MXCHECK_EQ(op->args.size(), 2U);
      uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
      uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
      uint64_t val = (high << 32U) | low;
      PrintUIntConst(op->dtype, val, os, this);
    } else if (op->op.same_as(builtin::bitwise_xor())) {
      PrintBinaryIntrinsic(op, " ^ ", os, this);
    } else if (op->op.same_as(builtin::bitwise_or())) {
      PrintBinaryIntrinsic(op, " | ", os, this);
    } else if (op->op.same_as(builtin::bitwise_not())) {
      MXCHECK_EQ(op->args.size(), 1U);
      os << "(~";
      this->PrintExpr(op->args[0], os);
      os << ')';
    } else if (op->op.same_as(builtin::shift_left())) {
      PrintBinaryIntrinsic(op, " << ", os, this);
    } else if (op->op.same_as(builtin::shift_right())) {
      PrintBinaryIntrinsic(op, " >> ", os, this);
    } else if (op->op.same_as(builtin::if_then_else()) ||
               op->op.same_as(builtin::hlo_if_then_else())) {
      os << "(";
      PrintExpr(op->args[0], os);
      os << " ? ";
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    } else if (op->op.same_as(builtin::address_of())) {
      MXTHROW << "not support builtin::address_of";
      //      const LoadNode* l = op->args[0].as<LoadNode>();
      //      MXCHECK(op->args.size() == 1 && l);
      //      os << "((";
      //      this->PrintType(l->dtype.element_of(), os);
      //      os << " *)" << this->GetVarID(l->buffer_var.get()) << " + "
      //         << "(";
      //      this->PrintExpr(l->index, os);
      //      if (l->dtype.bits() == 4 || (l->dtype.bits() == 1 && l->dtype.is_int())) {
      //        os << " / " << (32 / l->dtype.bits());
      //      }
      //      os << "))";
    } else if (op->op.same_as(builtin::isnullptr())) {
      MXCHECK_EQ(op->args.size(), 1U);
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " == NULL)";
    } else if (op->op.same_as(builtin::reinterpret())) {
      int ssa_scope = BeginScope();
      String rhs = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype, os);
      os << "(*(";
      this->PrintType(op->dtype, os);
      os << " *)(&(" << rhs << ")))";
      EndScope(ssa_scope);
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " != ";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      MXLOG(FATAL) << "Unresolved call " << op->op;
    }
  } else {
    // MXCHECK(op->op.as<GlobalVarNode>());
    MXLOG(FATAL) << "Do not yet support cross function call";
    MXCHECK(false);
  }
}

void CodeGenC::PrintVecBinaryOp(const String& op,
                                DataType t,
                                PrimExpr lhs,
                                PrimExpr rhs,
                                std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os << "(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

void CodeGenC::VisitExpr_(const PrimLetNode* op, std::ostream& os) {  // NOLINT(*)
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    MXCHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  String value = PrintExpr(op->value);
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

// void CodeGenC::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
//  // constraint of current logic
//  MXCHECK_EQ(op->base.dtype(), DataType::Int(32));
//  os << "((int" << op->lanes << ")(";
//  for (int i = 0; i < op->lanes; i++) {
//    os << "(" << PrintExpr(op->base) << ")"
//       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
//    if (i != op->lanes - 1)
//      os << ", ";
//  }
//  os << "))";
//}

// void CodeGenC::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
//  LOG(FATAL) << "Shuffle: not supported ";
//}
//
// void CodeGenC::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
//  LOG(FATAL) << "Broadcast: not supported ";
//}

void CodeGenC::VisitExpr_(const PrimSelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

template <typename T>
inline void PrintHLOBinaryExpr(const T* op,
                               const char* opstr,
                               std::ostream& os,  // NOLINT(*)
                               CodeGenC* p) {
  if (isalpha(opstr[0])) {
    os << opstr << '(';
    p->PrintExpr(op->a, os);
    os << ", ";
    p->PrintExpr(op->b, os);
    os << ')';
  } else {
    os << '(';
    p->PrintExpr(op->a, os);
    os << ' ' << opstr << ' ';
    p->PrintExpr(op->b, os);
    os << ')';
  }
}
template <typename T>
inline void SwitchPrintHLOBinaryExpr(const T* op,
                                     const char* opstr1,
                                     const char* opstr2,
                                     std::ostream& os,  // NOLINT(*)
                                     CodeGenC* p) {
  const auto& a_ty = RemoveReference(op->a->checked_type_);
  const auto& b_ty = RemoveReference(op->b->checked_type_);
  if (!IsObjectType(a_ty) && a_ty == b_ty) {
    PrintHLOBinaryExpr(op, opstr1, os, p);
  } else {
    PrintHLOBinaryExpr(op, opstr2, os, p);
  }
}

void CodeGenC::VisitExpr_(const HLOAddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::add", os, this);
}
void CodeGenC::VisitExpr_(const HLOSubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::sub", os, this);
}
void CodeGenC::VisitExpr_(const HLOMulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::mul", os, this);
}
void CodeGenC::VisitExpr_(const HLOFloorDivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::floordiv", os, this);
}
void CodeGenC::VisitExpr_(const HLOFloorModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::floormod", os, this);
}
void CodeGenC::VisitExpr_(const HLOEqualNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, "==", "ArithOps::eq", os, this);
}
void CodeGenC::VisitExpr_(const HLONotEqualNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, "!=", "ArithOps::ne", os, this);
}
void CodeGenC::VisitExpr_(const HLOLessThanNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, "<", "ArithOps::lt", os, this);
}
void CodeGenC::VisitExpr_(const HLOLessEqualNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, "<=", "ArithOps::le", os, this);
}
void CodeGenC::VisitExpr_(const HLOGreaterThanNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, ">", "ArithOps::gt", os, this);
}
void CodeGenC::VisitExpr_(const HLOGreaterEqualNode* op, std::ostream& os) {  // NOLINT(*)
  SwitchPrintHLOBinaryExpr(op, ">=", "ArithOps::ge", os, this);
}
void CodeGenC::VisitExpr_(const HLOAndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::And", os, this);
}
void CodeGenC::VisitExpr_(const HLOOrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintHLOBinaryExpr(op, "ArithOps::Or", os, this);
}
void CodeGenC::VisitExpr_(const HLONotNode* op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenC::VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) {
  PrintIndent(os);
  if (op->var->IsInstance<PrimVarNode>()) {
    auto var = Downcast<PrimVar>(op->var);
    var_idmap_[var.get()] = var->name_hint.operator String();
    this->PrintType(var->dtype, os);
    os << " " << var->name_hint;
  } else if (op->var->IsInstance<HLOVarNode>()) {
    auto var = Downcast<HLOVar>(op->var);
    var_idmap_[var.get()] = var->name_hint().operator String();
    this->PrintType(var->type_annotation, os);
    os << " " << var->name_hint();
  } else {
    MXCHECK(false) << "only support PrimVar and HLOVar...";
  }
  if (op->init_value.defined()) {
    os << " = (";
    VisitExpr(op->init_value, os);
    os << ");";
  } else {
    os << ";";
  }
  PrintSpanWithNewLine(op->span, os);
}

void CodeGenC::VisitStmt_(const AssignStmtNode* op, std::ostream& os) {
  String value = PrintExpr(op->rhs);
  PrintIndent(os);
  os << PrintExpr(op->lhs) << " = " << value << ";";
  PrintSpanWithNewLine(op->span, os);
}

void CodeGenC::VisitStmt_(const ReturnStmtNode* op, std::ostream& os) {
  String value = PrintExpr(op->value);
  PrintIndent(os);
  // TODO: fix type_infer bug and use move
  if (false && op->value->checked_type()->IsInstance<ObjectTypeNode>() &&
      !current_func_rt_type_->IsInstance<PrimTypeNode>() &&
      !current_func_rt_type_->IsInstance<ObjectTypeNode>()) {
    os << "return (" << value << ").MoveToObjectRef<";
    PrintType(current_func_rt_type_, os);
    os << ">();";
  } else {
    os << "return (" << value << ");";
  }
  PrintSpanWithNewLine(op->span, os);
}

void CodeGenC::VisitStmt_(const AssertStmtNode* op, std::ostream& os) {
  auto py_info = this->GenPythonStyleSpanMessage(op->span, this->current_py_func_name_);
  String cond = PrintExpr(op->condition);
  PrintIndent(os);
  os << "if (!(" << cond << ")) { THROW_PY_AssertionError(" << py_info;
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    os << ", \"" << str->value << "\"";
  }
  os << "); }";
  PrintSpanWithNewLine(op->span, os);
  this->PrintStmt(op->body, os);
}

void CodeGenC::VisitStmt_(const ForNode* op, std::ostream& os) {
  int step_state = -2;  // -1 for lt 0; 1 for gt 0
  if (auto* step_node = op->step.as<IntImmNode>()) {
    step_state = step_node->value >= 0 ? 1 : -1;
  }
  String min = PrintExpr(op->min);
  String max = PrintExpr(op->max);
  String step = PrintExpr(op->step);
  PrintIndent(os);
  String vid, t_vid;
  if (op->yield_mode) {
    vid = GetVarID(op->loop_var.get());
    t_vid = GetVarID(op->tmp_loop_var.get());
  } else {
    vid = AllocVarID(op->loop_var.get());
    t_vid = AllocVarID(op->tmp_loop_var.get());
  }
  os << "for (";
  if (!op->yield_mode) {
    PrintType(op->loop_var.dtype(), os);
    os << " ";
  }
  os << t_vid << " = " << min << "; ";
  if (step_state == 1) {
    os << t_vid << " < " << max << "; ";
  } else if (step_state == -1) {
    os << t_vid << " > " << max << "; ";
  } else {
    os << step << " > 0 ? (";
    os << t_vid << " < " << max << ") : ";
    os << t_vid << " > " << max << "; ";
  }
  os << t_vid << " += " << step << ") {";
  this->PrintSpanWithNewLine(op->span, os);

  int for_scope = BeginScope();
  PrintIndent(os);
  if (!op->yield_mode) {
    PrintType(op->loop_var.dtype(), os);
    os << ' ';
  }
  os << vid << " = " << t_vid << ";";
  this->PrintSpanWithNewLine(op->span, os);
  PrintStmt(op->body, os);
  this->EndScope(for_scope);
  PrintIndent(os);
  os << "}\n";
}

void CodeGenC::VisitStmt_(const AutoForNode* op, std::ostream& os) {
  auto py_info = this->GenPythonStyleSpanMessage(op->span, this->current_py_func_name_);
  struct VarInfo {
    String name;
    String type;
    Type ir_type;
  };
  auto FuncGetVarReprs = [this, op](const Array<BaseExpr>& expr_arr) -> std::vector<VarInfo> {
    std::vector<VarInfo> results;
    for (auto& local_var : expr_arr) {
      std::stringstream ss;
      PrintType(local_var->checked_type(), ss);
      String local_var_ty = ss.str();
      if (op->yield_mode) {
        results.push_back(VarInfo{GetVarID(local_var),
                                  std::move(local_var_ty),
                                  RemoveReference(local_var->checked_type())});
      } else {
        results.push_back(VarInfo{AllocVarID(local_var),
                                  std::move(local_var_ty),
                                  RemoveReference(local_var->checked_type())});
      }
    }
    return results;
  };

  std::vector<VarInfo> loop_var_reprs = FuncGetVarReprs(op->loop_vars);
  std::vector<VarInfo> loop_var_holder_reprs = FuncGetVarReprs(op->loop_vars_holder);
  std::vector<VarInfo> iter_var_reprs = FuncGetVarReprs(op->iter_vars);
  std::vector<VarInfo> iter_end_var_reprs = FuncGetVarReprs(op->iter_end_vars);
  std::vector<VarInfo> eval_cons_reprs = FuncGetVarReprs(op->eval_containers);
  String raw_container = PrintExpr(op->raw_container);
  std::unordered_map<String, VarInfo> temp_vars;
  for (auto& temp_var : op->temp_vars) {
    String var_key = temp_var.first;
    String var_name;
    if (op->yield_mode) {
      var_name = GetVarID(temp_var.second);
    } else {
      var_name = AllocVarID(temp_var.second);
    }
    std::stringstream ss;
    PrintType(temp_var.second->checked_type(), ss);
    String var_type = ss.str();
    temp_vars.emplace(var_key, VarInfo{var_name, var_type, temp_var.second->checked_type()});
  }
  MXCHECK(op->raw_container->checked_type()->Iterable()) << raw_container << " is not iterable";

  // unroll zip
  bool unroll_zip_state = [op]() {
    if (auto* zip_node = op->raw_container.as<HLOZipNode>()) {
      if (zip_node->values.size() == op->loop_vars.size()) {
        return true;
      }
    }
    return false;
  }();
  // unroll enumerate
  bool unroll_enumerate_state = [op]() {
    if (auto* enum_node = op->raw_container.as<HLOEnumerateNode>()) {
      if (2 == op->loop_vars.size()) {
        return true;
      }
    }
    return false;
  }();
  bool value_is_std_tuple = false;
  if (op->raw_container.as<HLOEnumerateNode>() || op->raw_container.as<HLOZipNode>()) {
    value_is_std_tuple = !(unroll_zip_state || unroll_enumerate_state);
  }

  /**
   * {
   *   // step1: eval iterable expr
   *   auto const& eval_cons_i = container_i;
   *   ... // unroll zip
   *
   *   // step2: make iter var
   *   HAS_BEGIN_END:
   *     auto iter_var_i = eval_cons_i.begin(); // has begin end
   *   ELSE:
   *     auto iter_var_j = Kernel_Iterable::make(eval_cons_j); // generic
   *     iter_var_j_has_next &= iter_var_j.HasNext(); // generic
   *
   *   // step3: make loop
   *   while (iter_var_i_has_next && (iter_var_j != eval_cons_i.end()) && ...)
   *   {
   *
   *   // step4: make loop var
   *   UNROLL OR LoopVarNum=1:
   *     HAS_BEGIN_END:
   *       Type loop_var_i = *iter_var_i;
   *       ++iter_var_i;
   *     ELSE:
   *       Type loop_var_j = iter_var_j.Next(&vid_iter_has_next);
   *   ELSE:
   *     HAS_BEGIN_END:
   *       Type value_var = *iter_var;
   *       ++iter_var;
   *     ELSE:
   *       Type value_var = iter_var.Next(&vid_iter_has_next);
   *     MXCHEK(kernel_builtins_len(value_var) == loop_var_num);
   *     Type loop_var_i = kernel_builtins_unpack<i>(value_var);
   *     ... //
   *     {
   *       // step5: body
   *       body;
   *     }
   *   }
   * }
   */

  // scope for eval expr to be iterated
  PrintIndent(os);
  os << "{\n";
  int for_scope_eval_container_expr = BeginScope();

  // step1: eval iterable expr
  {
    if (unroll_zip_state) {
      auto* zip_node = op->raw_container.as<HLOZipNode>();
      for (auto zi = 0; zi < zip_node->values.size(); ++zi) {
        String zip_arg_i_repr = PrintExpr(zip_node->values[zi]);
        PrintIndent(os);
        if (!op->yield_mode) {
          os << "auto const& ";
        }
        os << eval_cons_reprs[zi].name << " = " << zip_arg_i_repr << ";";
        PrintSpanWithNewLine(op->span, os);
      }
    } else if (unroll_enumerate_state) {
      auto* enum_node = op->raw_container.as<HLOEnumerateNode>();
      String enum_arg_repr = PrintExpr(enum_node->value);
      PrintIndent(os);
      if (!op->yield_mode) {
        os << "auto const& ";
      }
      os << eval_cons_reprs[0].name << " = " << enum_arg_repr << ";";
      PrintSpanWithNewLine(op->span, os);
    } else {
      PrintIndent(os);
      if (!op->yield_mode) {
        if (auto* ptr = op->raw_container.as<HLOEnumerateNode>()) {
          os << "auto const&";
        } else if (auto* ptr = op->raw_container.as<HLOZipNode>()) {
          os << "auto const&";
        } else {
          PrintType(op->raw_container->checked_type(), os);
          if (!IsRefType(op->raw_container->checked_type())) {
            os << " const& ";
          }
        }
      }
      os << eval_cons_reprs[0].name << " = " << raw_container << ";";
      PrintSpanWithNewLine(op->span, os);
    }
  }

  // step2: make iter var
  {
    // cache loop var holder
    if (!op->yield_mode) {
      for (auto& loop_var_holder_repr : loop_var_holder_reprs) {
        PrintIndent(os);
        os << loop_var_holder_repr.type << " " << loop_var_holder_repr.name << ";";
        PrintSpanWithNewLine(op->span, os);
      }
    }
    // cache enumerate pos
    if (unroll_enumerate_state) {
      MXCHECK(op->temp_vars.contains(AutoFor::TEMP_ENUMERATE_POS_VAR_KEY));
      auto* enum_node_ptr = op->raw_container.as<HLOEnumerateNode>();
      String enum_pos_vid = temp_vars[AutoFor::TEMP_ENUMERATE_POS_VAR_KEY].name;
      PrintIndent(os);
      if (!op->yield_mode) {
        os << temp_vars[AutoFor::TEMP_ENUMERATE_POS_VAR_KEY].type << " ";
      }
      os << enum_pos_vid << " = ";
      PrintExpr(enum_node_ptr->start, os);
      os << ";";
      PrintSpanWithNewLine(op->span, os);
    }
    MXCHECK_EQ(iter_var_reprs.size(), iter_end_var_reprs.size());
    for (auto ii = 0; ii < iter_var_reprs.size(); ++ii) {
      bool has_begin_end = op->eval_containers[ii]->checked_type()->HasBeginEnd();
      // print begin or make_iterable
      PrintIndent(os);
      if (!op->yield_mode) {
        os << "auto ";
      }
      if (has_begin_end) {
        os << iter_var_reprs[ii].name << " = " << eval_cons_reprs[ii].name << ".begin();";
      } else {
        if (IsIteratorType(op->eval_containers[ii])) {
          os << iter_var_reprs[ii].name << " = " << eval_cons_reprs[ii].name << ";";
        } else {
          os << iter_var_reprs[ii].name << " = Kernel_Iterable::make(" << eval_cons_reprs[ii].name
             << ");";
        }
      }
      PrintSpanWithNewLine(op->span, os);
      // print end or has_next
      PrintIndent(os);
      if (!op->yield_mode) {
        os << "auto ";
      }
      if (has_begin_end) {
        os << iter_end_var_reprs[ii].name << " = " << eval_cons_reprs[ii].name << ".end();";
      } else {
        os << iter_end_var_reprs[ii].name << " = " << iter_var_reprs[ii].name << ".HasNext();";
      }
      PrintSpanWithNewLine(op->span, os);
    }
  }

  // step3: make loop
  {
    bool not_first = false;
    PrintIndent(os);
    os << "while (";
    for (int ii = 0; ii < iter_var_reprs.size(); ++ii) {
      if (ii > 0) {
        os << " && ";
      }
      if (op->eval_containers[ii]->checked_type()->HasBeginEnd()) {
        os << "(" << iter_var_reprs[ii].name << " != " << iter_end_var_reprs[ii].name << ")";
      } else {
        os << iter_end_var_reprs[ii].name;
      }
    }
    os << ") {";
    PrintSpanWithNewLine(op->span, os);
  }
  int for_scope_loop = this->BeginScope();

  // step4: make loop var
  {
    ObjectType any_view_type(true, op->span);
    ObjectType any_value_type(false, op->span);
    auto IsItemNeedCast = [](bool from_is_view, const Type& target) {
      auto target_ty_node = RemoveReference(target).as<ObjectTypeNode>();
      return !(target_ty_node && (target_ty_node->is_view == from_is_view));
    };
    int li_start = 0;
    if (unroll_enumerate_state) {
      MXCHECK(op->temp_vars.contains(AutoFor::TEMP_ENUMERATE_POS_VAR_KEY));
      String enum_pos_vid = temp_vars[AutoFor::TEMP_ENUMERATE_POS_VAR_KEY].name;
      PrintIndent(os);
      if (!op->yield_mode) {
        os << loop_var_reprs[0].type << " ";
      }
      os << loop_var_reprs[0].name << " = " << enum_pos_vid << ";";
      PrintSpanWithNewLine(op->span, os);
      PrintIndent(os);
      os << "++" << enum_pos_vid << ";";
      PrintSpanWithNewLine(op->span, os);
      li_start = 1;
    }
    if (unroll_zip_state || unroll_enumerate_state || op->loop_vars.size() == 1) {
      int loop_var_holder_ii = 0;
      for (auto li = li_start; li < loop_var_reprs.size(); ++li) {
        auto iter_idx = li - li_start;
        PrintIndent(os);
        if (!op->yield_mode) {
          os << loop_var_reprs[li].type << " ";
        }
        const auto& cons_ty = RemoveReference(op->eval_containers[iter_idx]->checked_type_);
        if (cons_ty->HasBeginEnd()) {
          if (IsUnicodeType(cons_ty)) {
            auto* var_ty_n = RemoveReference(loop_var_reprs[li].ir_type).as<UnicodeTypeNode>();
            os << loop_var_reprs[li].name << " = ";
            if (var_ty_n && var_ty_n->is_view) {
              os << "unicode_view(" << iter_var_reprs[iter_idx].name << ", 1);";
            } else {
              os << "Unicode(1, *" << iter_var_reprs[iter_idx].name << ");";
            }
          } else if (cons_ty->IsFullTyped()) {
            os << loop_var_reprs[li].name << " = *" << iter_var_reprs[iter_idx].name << ";";
          } else {
            bool need_cast = IsItemNeedCast(false, loop_var_reprs[li].ir_type);
            if (IsListType(cons_ty) || IsTupleType(cons_ty) || IsSetType(cons_ty)) {
              os << loop_var_reprs[li].name << " = ";
              if (need_cast) {
                os << PrintTypeCast(any_value_type,
                                    loop_var_reprs[li].ir_type,
                                    "*" + iter_var_reprs[iter_idx].name,
                                    "the element in " + raw_container,
                                    py_info)
                   << ";";
              } else {
                os << "*" << iter_var_reprs[iter_idx].name << ";";
              }
            } else if (IsDictType(cons_ty)) {
              os << loop_var_reprs[li].name << " = ";
              if (need_cast) {
                os << PrintTypeCast(any_value_type,
                                    loop_var_reprs[li].ir_type,
                                    "(" + iter_var_reprs[iter_idx].name + ")->first",
                                    "the key of " + raw_container,
                                    py_info)
                   << ";";
              } else {
                os << "(" << iter_var_reprs[iter_idx].name << ")->first;";
              }
            } else {
              os << loop_var_reprs[li].name << " = *" << iter_var_reprs[iter_idx].name << ";";
            }
          }
          PrintSpanWithNewLine(op->span, os);
          PrintIndent(os);
          os << "++" << iter_var_reprs[iter_idx].name << ";";
          PrintSpanWithNewLine(op->span, os);
        } else {
          MXCHECK(loop_var_holder_ii < loop_var_holder_reprs.size()) << "internal error";
          auto loop_var_holder = loop_var_holder_reprs[loop_var_holder_ii++];
          String vid_iter_has_next = iter_end_var_reprs[iter_idx].name;
          String vid_iter_next_holder = loop_var_holder.name;
          auto* loop_var_type_li = RemoveReference(loop_var_reprs[li].ir_type).as<ObjectTypeNode>();
          if (value_is_std_tuple || (loop_var_type_li && !loop_var_type_li->is_view)) {
            os << loop_var_reprs[li].name << " = " << iter_var_reprs[iter_idx].name << ".Next(&"
               << vid_iter_has_next << ");";
          } else {
            bool need_cast = IsItemNeedCast(true, loop_var_reprs[li].ir_type);
            os << loop_var_reprs[li].name << " = ";
            std::stringstream rhs_value;
            rhs_value << iter_var_reprs[iter_idx].name << ".NextView(&" << vid_iter_has_next
                      << ", &" << vid_iter_next_holder << ")";
            if (need_cast) {
              os << PrintTypeCast(any_view_type,
                                  loop_var_reprs[li].ir_type,
                                  rhs_value.str(),
                                  "the next element in " + raw_container,
                                  py_info);
            } else {
              os << rhs_value.str();
            }
            os << ";";
          }
          PrintSpanWithNewLine(op->span, os);
        }
      }
    } else {
      MXCHECK(op->temp_vars.contains(AutoFor::TEMP_VALUE_VAR_KEY));
      const String& vid_value = temp_vars[AutoFor::TEMP_VALUE_VAR_KEY].name;
      const String& vid_type = temp_vars[AutoFor::TEMP_VALUE_VAR_KEY].type;
      const Type& vid_ir_type = RemoveReference(temp_vars[AutoFor::TEMP_VALUE_VAR_KEY].ir_type);
      // cache tmp value
      PrintIndent(os);
      if (!op->yield_mode) {
        os << "const auto& ";
      }
      const auto& cons_ty = RemoveReference(op->eval_containers[0]->checked_type_);
      if (cons_ty->HasBeginEnd()) {
        if (IsUnicodeType(cons_ty)) {
          auto* vid_ir_ty_n = vid_ir_type.as<UnicodeTypeNode>();
          if (vid_ir_ty_n && vid_ir_ty_n->is_view) {
            os << vid_value << " = unicode_view(" << iter_var_reprs[0].name << ", 1);";
          } else {
            os << vid_value << " = Unicode(1, *" << iter_var_reprs[0].name << ");";
          }
        } else if (cons_ty->IsFullTyped()) {
          os << vid_value << " = *" << iter_var_reprs[0].name << ";";
        } else {
          bool need_cast = IsItemNeedCast(false, vid_ir_type);
          if (IsListType(cons_ty) || IsTupleType(cons_ty) || IsSetType(cons_ty)) {
            if (need_cast) {
              os << vid_value << " = "
                 << PrintTypeCast(any_value_type,
                                  vid_ir_type,
                                  "*" + iter_var_reprs[0].name,
                                  "the element in " + raw_container,
                                  py_info)
                 << ";";
            } else {
              os << vid_value << " = *(" << iter_var_reprs[0].name << ");";
            }
          } else if (IsDictType(cons_ty)) {
            if (need_cast) {
              os << vid_value << " = "
                 << PrintTypeCast(any_value_type,
                                  vid_ir_type,
                                  "(" + iter_var_reprs[0].name + ")->first",
                                  "the key of " + raw_container,
                                  py_info)
                 << ";";
            } else {
              os << vid_value << " = (" << iter_var_reprs[0].name << ")->first;";
            }
          } else {
            os << vid_value << " = *" << iter_var_reprs[0].name << ";";
          }
        }
        PrintSpanWithNewLine(op->span, os);
        PrintIndent(os);
        os << "++" << iter_var_reprs[0].name << ";";
        PrintSpanWithNewLine(op->span, os);
      } else {
        MXCHECK(!loop_var_holder_reprs.empty()) << "internal error";
        auto loop_var_holder = loop_var_holder_reprs[0];
        String vid_iter_has_next = iter_end_var_reprs[0].name;
        String vid_iter_next_holder = loop_var_holder.name;
        os << vid_value << " = " << iter_var_reprs[0].name << ".NextView(&" << vid_iter_has_next
           << ", &" << vid_iter_next_holder << ");";
        PrintSpanWithNewLine(op->span, os);
      }
      // check unpack size
      PrintIndent(os);
      os << "if (kernel_builtins_len(" << vid_value << ") != " << op->loop_vars.size()
         << ") { THROW_PY_ValueError(" << py_info << ", \"values to unpack mismatch\"); }";
      PrintSpanWithNewLine(op->span, os);
      // unpack
      for (auto i = 0; i < op->loop_vars.size(); ++i) {
        PrintIndent(os);
        if (!op->yield_mode) {
          os << loop_var_reprs[i].type << " ";
        }
        os << loop_var_reprs[i].name << " = kernel_builtins_unpack<" << i << ", "
           << loop_var_reprs[i].type << ">(" << vid_value << ");";
        PrintSpanWithNewLine(op->span, os);
      }
    }
  }

  // step5:
  {
    PrintIndent(os);
    os << "{\n";
    int for_scope_body = BeginScope();
    PrintStmt(op->body, os);
    EndScope(for_scope_body);
    PrintIndent(os);
    os << "}\n";
  }
  EndScope(for_scope_loop);
  PrintIndent(os);
  os << "}\n";

  EndScope(for_scope_eval_container_expr);
  PrintIndent(os);
  os << "}\n";
}

void CodeGenC::VisitStmt_(const WhileNode* op, std::ostream& os) {
  String cond = PrintExpr(op->cond);
  PrintIndent(os);
  os << "while (" << cond << ") {";
  this->PrintSpanWithNewLine(op->span, os);
  int for_scope = BeginScope();
  PrintStmt(op->body, os);
  this->EndScope(for_scope);
  PrintIndent(os);
  os << "}\n";
}

void CodeGenC::VisitStmt_(const BreakNode* op, std::ostream& os) {
  PrintIndent(os);
  os << "break;";
  this->PrintSpanWithNewLine(op->span, os);
}

void CodeGenC::VisitStmt_(const ContinueNode* op, std::ostream& os) {
  PrintIndent(os);
  os << "continue;";
  this->PrintSpanWithNewLine(op->span, os);
}

void CodeGenC::VisitStmt_(const IfThenElseNode* op, std::ostream& os) {
  String cond = PrintExpr(op->condition);
  PrintIndent(os);
  os << "if (" << cond << ") {";
  this->PrintSpanWithNewLine(op->span, os);
  int then_scope = BeginScope();
  PrintStmt(op->then_case, os);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent(os);
    os << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case, os);
    this->EndScope(else_scope);
  }
  PrintIndent(os);
  os << "}\n";
}

void CodeGenC::VisitStmt_(const SeqStmtNode* op, std::ostream& os) {
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt, os);
  }
}

void CodeGenC::VisitStmt_(const ExprStmtNode* op, std::ostream& os) {
  this->PrintIndent(os);
  os << "(void)";
  VisitExpr(op->expr, os);
  os << ";";
  PrintSpanWithNewLine(op->span, os);
}

// high level expression
void CodeGenC::VisitExpr_(const HLOVarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}

void CodeGenC::VisitExpr_(const ConstructorNode* op, std::ostream& os) {  // NOLINT(*)
  os << "Kernel_" << op->name_hint << "::make";
  if (op->name_hint == "Tuple") {
    return;
  }
  if (!op->inputs.empty()) {
    os << "<";
  }
  for (size_t i = 0; i < op->inputs.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    PrintType(op->inputs[i], os);
  }
  if (!op->inputs.empty()) {
    os << ">";
  }
}

void CodeGenC::PrintAsRTValue(::matxscript::ir::BaseExpr expr, std::ostream& os) {
  if (auto* n = expr.as<StringImmNode>()) {
    VisitExpr_(n, os);
  } else if (auto* n = expr.as<IntImmNode>()) {
    VisitExpr_(n, os);
  } else if (auto* n = expr.as<FloatImmNode>()) {
    VisitExpr_(n, os);
  } else {
    VisitExpr(expr, os);
  }
}

void CodeGenC::VisitExpr_(const InitializerListNode* op, std::ostream& os) {
  PrintAsInitializeList(op, os);
}

void CodeGenC::PrintAsInitializeList(const InitializerListNode* op, std::ostream& os) {
  os << "{";
  if (op->fields.size() > 0) {
    os << "(";
  }
  for (int32_t i = 0; i < op->fields.size(); ++i) {
    PrintAsRTValue(op->fields[i], os);
    if (i + 1 != op->fields.size()) {
      os << "), (";
    }
  }
  if (op->fields.size() > 0) {
    os << ")";
  }
  os << "}";
}

void CodeGenC::PrintAsInitializeDict(const InitializerDictNode* op, std::ostream& os) {
  os << "{";
  if (op->fields.size() > 0) {
    os << "{";
  }
  int32_t i = 0;
  for (auto itr = op->fields.begin(); itr != op->fields.end(); ++itr) {
    PrintAsRTValue((*itr).first, os);
    os << ", ";
    PrintAsRTValue((*itr).second, os);
    if (i + 1 != op->fields.size()) {
      os << "}, {";
    }
    ++i;
  }
  if (op->fields.size() > 0) {
    os << "}";
  }
  os << "}";
}

void CodeGenC::PrintConstructorValueType(const ConstructorNode* op, std::ostream& os) {
  if (op->name_hint == "Tuple") {
    os << "Tuple::value_type";
    return;
  }
  os << op->name_hint;
  if (!op->inputs.empty()) {
    os << "<";
  }
  for (size_t i = 0; i < op->inputs.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    PrintType(op->inputs[i], os);
  }
  if (!op->inputs.empty()) {
    os << ">";
  }
  os << "::value_type";
}

void CodeGenC::PrintAsConstructor(const CallNode* op, std::ostream& os) {
  // adt constructor
  const auto* ptr_cons = op->op.as<ConstructorNode>();
  if (op->args.size() == 1 && op->args[0].as<HLOIteratorNode>()) {
    const auto* init_iter = op->args[0].as<HLOIteratorNode>();
    std::stringstream tmp_ss;
    PrintAsRTValue(init_iter->container, tmp_ss);
    auto iter_c_name = tmp_ss.str();
    if (init_iter->container->checked_type()->IsInstance<ObjectTypeNode>()) {
      VisitExpr_(ptr_cons, os);
      os << "(Kernel_Iterable::make(" << iter_c_name << "))";
    } else {
      os << ptr_cons->name_hint << "(";
      os << iter_c_name << ".begin(), " << iter_c_name << ".end()";
      os << ")";
    }
  } else {
    VisitExpr_(ptr_cons, os);
    os << "(";
    for (int32_t i = 0; i < op->args.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      if (const InitializerListNode* init_args = op->args[i].as<InitializerListNode>()) {
        os << "std::initializer_list<";
        PrintConstructorValueType(ptr_cons, os);
        os << ">";
        PrintAsInitializeList(init_args, os);
      } else if (const InitializerDictNode* init_kwargs = op->args[i].as<InitializerDictNode>()) {
        os << "std::initializer_list<";
        PrintConstructorValueType(ptr_cons, os);
        os << ">";
        PrintAsInitializeDict(init_kwargs, os);
      } else if (const HLOIteratorNode* init_iter = op->args[i].as<HLOIteratorNode>()) {
        std::stringstream tmp_ss;
        PrintAsRTValue(init_iter->container, tmp_ss);
        auto iter_c_name = tmp_ss.str();
        os << "::matxscript::runtime::Iterator(" << iter_c_name << ")";
      } else {
        os << "(";
        PrintAsRTValue(op->args[i], os);
        os << ")";
      }
    }
    os << ")";
  }
}

void CodeGenC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.as<ConstructorNode>()) {
    PrintAsConstructor(op, os);
  } else if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);
    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      MXCHECK_GE(op->args.size(), 1U);
      MXCHECK(op->type_args.empty()) << "[call extern] type_args is not supported!!!";
      auto func = Downcast<StringImm>(op->args[0]);
      this->PrintCallExtern(op->checked_type(), func->value, op->args, true, os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      this->PrintCallFunction(op, os);
    } else if (op_attr_method_symbol_.count(Downcast<Op>(op->op))) {
      // call method if the op itself have a method symbol.
      this->PrintCallMethod(op, os);
    } else if (op->op.same_as(builtin::call_lambda())) {  // hlo arith
      MXCHECK_GE(op->args.size(), 1U);
      // function
      this->VisitExpr(op->args[0], os);
      os << "(";
      for (size_t i = 1; i < op->args.size(); ++i) {
        if (i > 1) {
          os << ", ";
        }
        this->VisitExpr(op->args[i], os);
      }
      os << ")";
    } else if (op->op.same_as(builtin::torch_ops())) {
      MXCHECK(op->args.size() >= 1);
      auto func_name = op->args[0].as<StringImmNode>();
      MXCHECK(func_name != nullptr);
      os << "call_native_function(\"torch_ops_" << func_name->value << "\"";
      for (auto i = 1; i < op->args.size(); ++i) {
        os << ", ";
        this->PrintExpr(op->args[i], os);
      }
      os << ")";
    } else if (op->op.same_as(builtin::numpy_ops())) {
      MXCHECK(op->args.size() >= 1);
      auto func_name = op->args[0].as<StringImmNode>();
      MXCHECK(func_name != nullptr);
      os << "call_native_function(\"numpy_ops_" << func_name->value << "\"";
      for (auto i = 1; i < op->args.size(); ++i) {
        os << ", ";
        this->PrintExpr(op->args[i], os);
      }
      os << ")";
    } else if (op->op.same_as(builtin::make_kwargs_op())) {
      MXCHECK(op->args.size() % 2 == 0);
      os << "::matxscript::runtime::Kwargs({";
      for (auto i = 0; i < op->args.size(); i += 2) {
        if (i > 0) {
          os << ", ";
        }
        os << "{";
        auto arg_key = op->args[i].as<StringImmNode>();
        os << "\"" << arg_key->value << "\"";
        os << ", ";
        this->PrintExpr(op->args[i + 1], os);
        os << "}";
      }
      os << "})";
    } else if (op->op.same_as(builtin::if_then_else()) ||
               op->op.same_as(builtin::hlo_if_then_else())) {
      MXCHECK(op->args.size() == 3) << "internal error";
      os << "(";
      PrintExpr(op->args[0], os);
      os << " ? ";
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    } else {
      MXLOG(FATAL) << "Unresolved call " << op->op;
    }
  } else {
    this->PrintExpr(op->op, os);
    os << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      this->VisitExpr(op->args[i], os);
    }
    os << ")";
  }
}

void CodeGenC::PrintCallMethod(const CallNode* op, std::ostream& os) {
  MXCHECK_GE(op->args.size(), 1);
  Op builtin_op = Downcast<Op>(op->op);
  String method = op_attr_method_symbol_.get(builtin_op, "").operator String();

  if (!op->type_args.empty()) {
    method.append("<");
  }
  for (auto i = 0; i < op->type_args.size(); ++i) {
    if (i > 0) {
      method.append(", ");
    }
    if (auto* imm_val_ptr = op->type_args[i].as<IntImmNode>()) {
      method.append(std::to_string(imm_val_ptr->value));
    } else {
      std::stringstream ty_ss;
      PrintType(runtime::Downcast<Type>(op->type_args[i]), ty_ss);
      method.append(ty_ss.str());
    }
  }
  if (!op->type_args.empty()) {
    method.append(">");
  }

  os << "(";
  VisitExpr(op->args[0], os);
  os << ")." << method << "(";
  size_t arg_index = 1;
  if (op->op.same_as(builtin::user_data_call_attr())) {
    // user data dispatch
    arg_index += 1;
    MXCHECK(op->args[1]->IsInstance<StringImmNode>())
        << "generic dispatch second arg must be function name";
    os << "\"" << op->args[1].as<StringImmNode>()->value << "\"";
    if (op->args.size() > arg_index) {
      os << ", ";
    }
  }
  for (size_t i = arg_index; i < op->args.size(); ++i) {
    if (i > arg_index) {
      os << ", ";
    }
    if (op->args[i].as<InitializerListNode>()) {
      PrintAsRTValue(op->args[i], os);
    } else {
      // VisitExpr(op->args[i], os);
      os << "(";
      PrintAsRTValue(op->args[i], os);
      // VisitExpr(op->args[i], os);
      os << ")";
    }
  }
  os << ")";
}

void CodeGenC::PrintCallFunction(const CallNode* op, std::ostream& os) {
  Op builtin_op = Downcast<Op>(op->op);
  String func_name = op_attr_global_symbol_.get(builtin_op, "").operator String();
  MXCHECK(!func_name.empty());

  auto* ptr_op = op->op.as<OpNode>();
  // step1: check arguments number
  if (ptr_op->num_inputs >= 0) {
    if (ptr_op->num_inputs_max == ptr_op->num_inputs) {
      MXCHECK_GE(ptr_op->num_inputs, op->args.size())
          << "[CodeGenC::PrintCallFunction] arg num is mismatched, expect " << ptr_op->num_inputs
          << ", but get " << op->args.size() << ", op:" << func_name;
    } else if (ptr_op->num_inputs_max > 0) {
      MXCHECK(op->args.size() >= ptr_op->num_inputs && op->args.size() <= ptr_op->num_inputs_max)
          << "[CodeGenC::PrintCallFunction] arg num is mismatched, expect (" << ptr_op->num_inputs
          << "-" << ptr_op->num_inputs_max << ")"
          << ", but get " << op->args.size() << ", op:" << func_name;
    } else {
      // -1 means it is variable length
      MXCHECK_GE(op->args.size(), ptr_op->num_inputs)
          << "[CodeGenC::PrintCallFunction] arg num is mismatched, expect " << ptr_op->num_inputs
          << " or more, but get " << op->args.size() << ", op:" << func_name;
    }
  } else {
    // -1 means it is variable length
    if (op->op.same_as(builtin::object___dispatch__())) {
      MXCHECK_GE(op->args.size(), 2)
          << "[CodeGenC::PrintGenericBuiltinOp] arg num is mismatched, expect ge 2"
          << ", but get " << op->args.size() << ", op:" << func_name;
    }
  }

  // step2: add typed arguments
  if (!op->type_args.empty()) {
    func_name.append("<");
  }
  for (auto i = 0; i < op->type_args.size(); ++i) {
    if (i > 0) {
      func_name.append(", ");
    }
    if (auto* imm_val_ptr = op->type_args[i].as<IntImmNode>()) {
      func_name.append(std::to_string(imm_val_ptr->value));
    } else {
      std::stringstream ty_ss;
      PrintType(runtime::Downcast<Type>(op->type_args[i]), ty_ss);
      func_name.append(ty_ss.str());
    }
  }
  if (!op->type_args.empty()) {
    func_name.append(">");
  }

  // step3: print function name
  os << func_name << "(";

  // step4: print arguments
  bool is_object = func_name.substr(0, strlen("kernel_object_")) == "kernel_object_";
  int32_t arg_index = 0;
  if (is_object) {
    MXCHECK(op->args.size() >= 1) << "object.func(...) first arg must be self";
    arg_index += 1;
    auto& self_type = RemoveReference(op->args[0]->checked_type_);
    if (self_type.as<ClassTypeNode>()) {
      os << "(";
      this->VisitExpr(op->args[0], os);
      os << ").operator RTView()";
    } else {
      this->VisitExpr(op->args[0], os);
    }
    if (ptr_op->num_inputs < 0 || op->args.size() > 1) {
      os << ", ";
    }
  }
  if (op->op.same_as(builtin::object___dispatch__())) {
    // user data dispatch
    arg_index += 1;
    MXCHECK(op->args[1]->IsInstance<StringImmNode>())
        << "generic dispatch second arg must be function name";
    os << "\"" << op->args[1].as<StringImmNode>()->value << "\"";
    os << ", ";
  } else if (op->op.same_as(builtin::object___getattr__()) ||
             op->op.same_as(builtin::object___setattr__())) {
    // user data dispatch
    arg_index += 1;
    MXCHECK(op->args[1]->IsInstance<StringImmNode>())
        << "generic dispatch second arg must be attribute name";
    os << "\"" << op->args[1].as<StringImmNode>()->value << "\"";
    if (op->args.size() > 2) {
      os << ", ";
    }
  }

  if (ptr_op->num_inputs < 0 && !op->op.same_as(builtin::builtins_print())) {
    os << "{";
  }
  for (int32_t i = arg_index; i < op->args.size(); ++i) {
    if (i != arg_index) {
      os << ", ";
    }
    this->VisitExpr(op->args[i], os);
  }
  if (ptr_op->num_inputs < 0 && !op->op.same_as(builtin::builtins_print())) {
    os << "}";
  }
  os << ")";
}

void CodeGenC::VisitExpr_(const EnumAttrNode* op, std::ostream& os) {
  os << op->enum_str;
}

void CodeGenC::VisitExpr_(const HLOCastNode* op, std::ostream& os) {  // NOLINT(*)
  auto py_info = this->GenPythonStyleSpanMessage(op->span, this->current_py_func_name_);
  std::stringstream value;
  this->PrintExpr(op->value, value);
  auto value_repr = value.str();

  os << PrintTypeCast(op->value->checked_type_, op->checked_type_, value_repr, value_repr, py_info);
}

void CodeGenC::VisitExpr_(const HLOMoveNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << "std::move(" << value.str() << ")";
}

void CodeGenC::VisitExpr_(const HLOEnumerateNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  std::stringstream start;
  this->PrintExpr(op->start, start);
  if (op->value->checked_type()->HasBeginEnd()) {
    os << "enumerate(" << value.str() << ", " << start.str() << ")";
  } else {
    os << "generic_enumerate(Kernel_Iterable::make(" << value.str() << "), " << start.str() << ")";
  }
}

void CodeGenC::VisitExpr_(const HLOZipNode* op, std::ostream& os) {  // NOLINT(*)
  bool has_begin_end = op->checked_type()->HasBeginEnd();
  std::stringstream values_os;
  for (auto i = 0; i < op->values.size(); ++i) {
    if (i > 0) {
      values_os << ", ";
    }
    if (!has_begin_end) {
      values_os << "Kernel_Iterable::make(";
    }
    this->PrintExpr(op->values[i], values_os);
    if (!has_begin_end) {
      values_os << ")";
    }
  }

  if (has_begin_end) {
    os << "builtins_zip(" << values_os.str() << ")";
  } else {
    os << "generic_builtins_zip(" << values_os.str() << ")";
  }
}

void CodeGenC::PrintStmt(const Stmt& n, std::ostream& os) {
  VisitStmt(n, os);
}

}  // namespace codegen
}  // namespace matxscript
