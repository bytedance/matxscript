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
 * \file codegen_c_host.cc
 */
#include "codegen_c_host.h"

#include <string>
#include <vector>

#include <matxscript/ir/module.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/func_registry_names_io.h>
#include <matxscript/runtime/function_name_rules.h>
#include <matxscript/runtime/module.h>
#include <matxscript/runtime/str_escape.h>
#include "assign_optimizer.h"
#include "binary_add_optimizer.h"
#include "caster_optimizer.h"
#include "func_args_optimizer.h"
#include "fuse_cont_get_set_item.h"
#include "move_optimizer.h"
#include "var_detect.h"
#include "yield_detect.h"

namespace matxscript {
namespace codegen {

CodeGenCHost::CodeGenCHost() {
  module_name_ = GetUniqueName("__matx_module_ctx");
}

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts) {
  emit_asserts_ = emit_asserts;
  declared_globals_.clear();
  decl_stream << "#include \"matxscript/runtime/codegen_all_includes.h\"\n";
  decl_stream << "#include <math.h>\n";
  decl_stream << "\nusing namespace ::matxscript::runtime;\n";
  decl_stream << "extern \"C\" void* " << symbol::library_module_ctx << " = NULL;\n\n";
  decl_stream << "extern \"C\" MATX_DLL MATXScriptFuncRegistry " << symbol::library_func_registry
              << ";\n\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenCHost::InitTypeRegistry(const ClassStmt& cls_stmt) {
  auto class_name = cls_stmt->name;
  this->stream << "extern \"C\" MATX_DLL MATXScriptFuncRegistry " << symbol::library_func_registry
               << class_name << ";\n";
}

void CodeGenCHost::BeginAnonymousNamespace() {
  this->stream << "namespace {\n";
}

void CodeGenCHost::EndAnonymousNamespace() {
  this->stream << "\n} // namespace\n\n";
}

void CodeGenCHost::AddUserStructDeclaration(const ClassStmt& cls_stmt) {
  auto class_name = cls_stmt->name;
  this->PrintIndent(this->stream);
  this->stream << "// User class forward declarations\n";
  this->PrintIndent(this->stream);
  this->stream << "struct " << class_name << ";\n";
  this->PrintIndent(this->stream);
  this->stream << "struct " << FunctionNameRules::get_class_view_name(class_name) << ";\n\n";
}

void CodeGenCHost::AddUserStructInitDeclaration(const ClassStmt& cls_stmt,
                                                const BaseFunc& init_func) {
  this->InitAllState();
  auto class_name = cls_stmt->name;
  auto class_view_name = FunctionNameRules::get_class_view_name(class_name);
  // init function wrapper
  if (init_func.defined()) {
    auto raw_params = init_func->GetParams();
    Array<BaseExpr> params;
    params.reserve(raw_params.size());
    for (size_t i = 1; i < raw_params.size(); ++i) {
      params.push_back(raw_params[i]);
    }
    this->PrintIndent(this->stream);
    auto wrapper_func = FunctionNameRules::add_wrapper_suffix(init_func->GetGlobalName());
    this->stream << class_view_name << " " << wrapper_func << "(";
    this->PrintLineVars(
        this->stream, params, init_func->GetDefaultParams(), true, true, true, true);
    stream << ");\n";
    this->stream << "int " << FunctionNameRules::add_packed_suffix(wrapper_func)
                 << "(MATXScriptAny*, int, MATXScriptAny*, void*);\n";
  } else {
    auto init_func_name = FunctionNameRules::add_class_prefix(class_name, "__init__");
    auto wrapper_func = FunctionNameRules::add_wrapper_suffix(init_func_name);
    this->PrintIndent(this->stream);
    this->stream << class_view_name << " " << wrapper_func << "();\n";
    this->stream << "int " << FunctionNameRules::add_packed_suffix(wrapper_func)
                 << "(MATXScriptAny*, int, MATXScriptAny*, void*);\n";
  }
}

static bool IsClassOnlyWithInitFunctions(const ClassType& cls_ty, const BaseFunc& init_func) {
  if (cls_ty->func_types.empty() || (init_func.defined() && cls_ty->func_types.size() == 1)) {
    return true;
  } else {
    return false;
  }
}

void CodeGenCHost::DefineUserStruct(const ClassStmt& cls_stmt,
                                    const std::unordered_map<String, BaseFunc>& methods) {
  this->InitAllState();
  auto class_name = cls_stmt->name;
  auto cls_ty = cls_stmt->type;

  String reserved_keyword = "_2_71828182846";
  auto virtual_var_name_tables = cls_ty->GetVarNamesLookupTable();
  auto virtual_var_types_tables = cls_ty->GetVarTypesLookupTable();
  auto GetNestedBaseHeaders = [](const Type& base) -> std::vector<StringRef> {
    std::vector<StringRef> all_headers;
    std::function<void(const Type& base)> fn;
    fn = [&](const Type& base) -> void {
      if (base.defined()) {
        auto base_node = base.as<ClassTypeNode>();
        MXCHECK(base_node) << "class base type can only be class, but get " << base;
        all_headers.push_back(base_node->header->name_hint);
        fn(base_node->base);
      }
    };
    fn(base);
    return all_headers;
  };

  // define user class
  this->PrintIndent(this->stream);
  String base_class = "IUserDataRoot";
  if (cls_ty->base.defined()) {
    auto base_node = cls_ty->base.as<ClassTypeNode>();
    MXCHECK(base_node) << "class base type can only be class, but get " << cls_ty->base;
    base_class = base_node->header->name_hint.operator String();
  }
  this->stream << "struct " << class_name << " : public " << base_class << " {\n";

  auto cls_scope = this->BeginScope();

  this->PrintIndent(this->stream);
  this->stream << "// flags for convert check\n";
  this->PrintIndent(this->stream);
  this->stream << "static uint32_t tag_s" << reserved_keyword << "_;\n";
  this->PrintIndent(this->stream);
  this->stream << "static uint32_t var_num_s" << reserved_keyword << "_;\n";
  this->PrintIndent(this->stream);
  this->stream << "static string_view class_name_s" << reserved_keyword << "_;\n";

  this->PrintIndent(this->stream);
  this->stream << "static IUserDataRoot::__FunctionTable__ function_table_s" << reserved_keyword
               << "_;\n";
  this->stream << "\n";

  this->PrintIndent(this->stream);
  this->stream << "// override meta functions\n";
  this->PrintIndent(this->stream);
  this->stream << "const char* ClassName" << reserved_keyword << "() const override { return \""
               << class_name << "\"; }\n";
  this->PrintIndent(this->stream);
  this->stream << "uint32_t tag" << reserved_keyword << "() const override { return tag_s"
               << reserved_keyword << "_; }\n";
  this->PrintIndent(this->stream);
  this->stream << "uint32_t size" << reserved_keyword << "() const override { return var_num_s"
               << reserved_keyword << "_; }\n\n";
  {
    // check isinstance
    auto all_nested_headers = GetNestedBaseHeaders(cls_ty);
    this->PrintIndent(this->stream);
    this->stream << "bool isinstance" << reserved_keyword << "(uint64_t tag) override {\n";
    auto isinstance_scope = this->BeginScope();
    this->PrintIndent(this->stream);
    this->stream << "static std::initializer_list<uint64_t> all_tags = {";
    for (size_t i = 0; i < all_nested_headers.size(); ++i) {
      if (i > 0) {
        this->stream << ", ";
      }
      this->stream << all_nested_headers[i] << "::tag_s" << reserved_keyword << "_";
    }
    this->stream << "};\n";
    this->PrintIndent(this->stream);
    this->stream << "return std::find(all_tags.begin(), all_tags.end(), tag) != all_tags.end();\n";
    this->EndScope(isinstance_scope);
    this->PrintIndent(this->stream);
    this->stream << "}\n\n";
  }

  this->PrintIndent(this->stream);
  this->stream << "std::initializer_list<string_view> VarNames" << reserved_keyword
               << "() const override {\n";
  auto var_names_scope = this->BeginScope();
  this->PrintIndent(this->stream);
  this->stream << "static std::initializer_list<string_view> __var_names_s__ = {";
  for (size_t i = 0; i < virtual_var_name_tables.size(); ++i) {
    auto var_name = virtual_var_name_tables[i];
    this->stream << "\"" << var_name << "\", ";
  }
  this->stream << "};\n";
  this->PrintIndent(this->stream);
  this->stream << "return __var_names_s__;\n";
  this->EndScope(var_names_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n\n";

  this->PrintIndent(this->stream);
  this->stream << "const ska::flat_hash_map<string_view, int64_t>& VarTable" << reserved_keyword
               << "() const override {\n";
  auto var_table_scope = this->BeginScope();
  this->PrintIndent(this->stream);
  this->stream << "static ska::flat_hash_map<string_view, int64_t> __var_table_s__ = {\n";
  auto var_table_data_scope = this->BeginScope();
  for (size_t i = 0; i < virtual_var_name_tables.size(); ++i) {
    auto var_name = virtual_var_name_tables[i];
    this->PrintIndent(this->stream);
    this->stream << "{\"" << var_name << "\", " << i << "}, \n";
  }
  this->EndScope(var_table_data_scope);
  this->PrintIndent(this->stream);
  this->stream << "};\n";
  this->PrintIndent(this->stream);
  this->stream << "return __var_table_s__;\n";
  this->EndScope(var_table_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n\n";

  this->PrintIndent(this->stream);
  this->stream << "// member vars\n";
  for (size_t i = 0; i < cls_ty->var_names.size(); ++i) {
    auto var_name = cls_ty->var_names[i];
    auto var_type = cls_ty->var_types[i];
    this->PrintIndent(this->stream);
    this->PrintType(var_type, this->stream);
    this->stream << " " << var_name;
    if ((!var_type->IsFullTyped()) &&
        (IsListType(var_type) || IsDictType(var_type) || IsSetType(var_type))) {
      this->stream << "{ObjectPtr<Object>{nullptr}}";
    }
    this->stream << ";\n";
  }
  this->stream << "\n";

  // Object pointer
  this->PrintIndent(this->stream);
  this->stream << "// Object pointer\n";
  this->PrintIndent(this->stream);
  this->stream << "Object* self_node_ptr_2_71828182846 = nullptr;\n";
  this->stream << "\n";

  // GetVar_2_71828182846
  this->PrintIndent(this->stream);
  this->stream << "// override GetVar_2_71828182846 functions\n";
  this->PrintIndent(this->stream);
  this->stream << "RTView GetVar_2_71828182846(int64_t idx) const override {\n";
  auto get_item_scope = this->BeginScope();

  this->PrintIndent(this->stream);
  this->stream << "switch (idx) {\n";
  for (size_t i = 0; i < virtual_var_name_tables.size(); ++i) {
    this->PrintIndent(this->stream);
    if (auto var_t_node = virtual_var_types_tables[i].as<ClassTypeNode>()) {
      std::stringstream ss_var;
      ss_var << virtual_var_name_tables[i] << ".operator RTView()";
      this->stream << "case " << i << ": { return " << ss_var.str() << "; } break;\n";
    } else {
      this->stream << "case " << i << ": { return " << virtual_var_name_tables[i] << "; } break;\n";
    }
  }
  this->PrintIndent(this->stream);
  this->stream << "default: { THROW_PY_IndexError(\"index overflow\"); return nullptr; } break;\n";
  this->stream << "\n";
  this->PrintIndent(this->stream);
  this->stream << "}\n";

  this->EndScope(get_item_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n";

  // SetVar_2_71828182846
  this->PrintIndent(this->stream);
  this->stream << "// override SetVar_2_71828182846 functions\n";
  this->PrintIndent(this->stream);
  this->stream << "void SetVar_2_71828182846(int64_t idx, const Any& val) override {\n";
  auto set_item_scope = this->BeginScope();

  this->PrintIndent(this->stream);
  this->stream << "switch (idx) {\n";
  for (size_t i = 0; i < virtual_var_name_tables.size(); ++i) {
    this->PrintIndent(this->stream);
    std::stringstream ss;
    auto var_type = virtual_var_types_tables[i];
    this->PrintType(var_type, ss);
    auto var_type_s = ss.str();

    if (auto var_t_node = var_type.as<ClassTypeNode>()) {
      auto var_cls_name = var_t_node->header->name_hint;
      this->stream << "case " << i << ": { this->" << virtual_var_name_tables[i]
                   << " = static_cast<" << var_type_s
                   << ">(MATXSCRIPT_TYPE_AS_V2(val, UserDataRef, \"" << var_cls_name
                   << "\")); } break;\n";
    } else {
      this->stream << "case " << i << ": { this->" << virtual_var_name_tables[i]
                   << " = MATXSCRIPT_TYPE_AS(val, " << var_type_s << ")"
                   << "; } break;\n";
    }
  }
  this->PrintIndent(this->stream);
  this->stream << "default: { THROW_PY_IndexError(\"index overflow\"); } break;\n";
  this->stream << "\n";
  this->PrintIndent(this->stream);
  this->stream << "}\n";

  this->EndScope(set_item_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n\n";

  // virtual functions
  this->PrintIndent(this->stream);
  this->stream << "// virtual methods\n";
  for (size_t i = 0; i < cls_ty->func_names.size(); ++i) {
    auto fn_type = cls_ty->func_types[i];
    String fn_name = cls_ty->func_names[i];
    String unbound_fn_name = cls_ty->unbound_func_names[i];
    auto itr_fn = methods.find(unbound_fn_name);
    MXCHECK(itr_fn != methods.end());
    auto base_func = itr_fn->second;
    this->PrintIndent(this->stream);
    this->stream << "virtual ";
    PrintType(fn_type->ret_type, this->stream);
    this->stream << " " << fn_name << "(";
    // skip first arg: 'self'
    PrintLineVars(this->stream,
                  base_func->GetParams(),
                  base_func->GetDefaultParams(),
                  true,
                  true,
                  true,
                  true,
                  false,
                  false,
                  true);
    stream << ");\n";
  }

  this->EndScope(cls_scope);
  this->PrintIndent(this->stream);
  this->stream << "};\n\n";

  // static var
  this->PrintIndent(this->stream);
  this->stream << "// flags for convert check\n";
  this->PrintIndent(this->stream);
  this->stream << "uint32_t " << class_name << "::tag_s" << reserved_keyword
               << "_ = " << cls_ty->tag << ";\n";
  this->PrintIndent(this->stream);
  this->stream << "uint32_t " << class_name << "::var_num_s" << reserved_keyword
               << "_ = " << cls_ty->var_names.size() << ";\n";
  this->PrintIndent(this->stream);
  this->stream << "string_view " << class_name << "::class_name_s" << reserved_keyword << "_ = \""
               << class_name << "\";\n";

  this->PrintIndent(this->stream);
  this->stream << "IUserDataRoot::__FunctionTable__ " << class_name << "::function_table_s"
               << reserved_keyword << "_ = ";
  if (cls_ty->base.defined()) {
    auto nested_base_headers = GetNestedBaseHeaders(cls_ty->base);
    this->stream << "IUserDataRoot::JoinFuncTables" << reserved_keyword
                 << "({IUserDataRoot::InitFuncTable" << reserved_keyword << "(&"
                 << symbol::library_func_registry << class_name << ", \"" << class_name << "\")";
    for (auto& base_header : nested_base_headers) {
      this->stream << ", " << base_header << "::function_table_s" << reserved_keyword << "_";
    }
    this->stream << "});\n";
  } else {
    this->stream << "IUserDataRoot::InitFuncTable" << reserved_keyword << "(&"
                 << symbol::library_func_registry << class_name << ", \"" << class_name << "\");\n";
  }
  this->stream << "\n";

  // define user class view
  auto class_view_name = FunctionNameRules::get_class_view_name(class_name);
  this->PrintIndent(this->stream);
  this->stream << "struct " << class_view_name << ": public IUserDataSharedViewRoot {\n";
  auto cls_view_scope = this->BeginScope();

  this->PrintIndent(this->stream);
  this->stream << "// member var\n";
  this->PrintIndent(this->stream);
  this->stream << class_name << " *ptr;\n";
  // this->PrintIndent(this->stream);
  // this->stream << "UserDataRef ud_ref{ObjectPtr<Object>(nullptr)};\n";

  this->PrintIndent(this->stream);
  this->stream << "// constructor\n";
  this->PrintIndent(this->stream);
  this->stream
      << class_view_name << "(" << class_name
      << " *ptr, UserDataRef ref) : ptr(ptr), IUserDataSharedViewRoot(std::move(ref)) {}\n";
  this->PrintIndent(this->stream);
  this->stream << class_view_name << "(" << class_name << " *ptr) : ptr(ptr) {}\n";
  this->PrintIndent(this->stream);
  this->stream << class_view_name << "() : ptr(nullptr) {}\n";
  this->PrintIndent(this->stream);
  this->stream << class_view_name << "(const matxscript::runtime::Any& ref) : " << class_view_name
               << "(MATXSCRIPT_TYPE_AS_V2(ref, UserDataRef, \"" << class_name << "\")) {}\n";

  this->PrintIndent(this->stream);
  this->stream << "// UserDataRef\n";
  this->PrintIndent(this->stream);
  this->stream << class_view_name << "(UserDataRef ref) {\n";
  auto cls_view_sub_scope = this->BeginScope();

  this->PrintIndent(this->stream);
  this->stream << "IUserDataRoot* base_ud_ptr = static_cast<IUserDataRoot*>";
  this->stream << "(ref.check_codegen_ptr(\"" << class_name << "\"));\n";

  this->PrintIndent(this->stream);
  this->stream << "if(!base_ud_ptr->isinstance" << reserved_keyword << "(" << class_name
               << "::tag_s" << reserved_keyword << "_)) {THROW_PY_TypeError(\"expect '"
               << class_name << "' but get '\", base_ud_ptr->ClassName" << reserved_keyword
               << "(), \"'\");}\n";
  this->PrintIndent(this->stream);
  this->stream << "ptr = static_cast<" << class_name << "*>(base_ud_ptr);\n";
  this->PrintIndent(this->stream);
  this->stream << "ud_ref = std::move(ref);\n";

  this->EndScope(cls_view_sub_scope);
  this->PrintIndent(this->stream);
  this->stream << "}\n";

  this->PrintIndent(this->stream);
  this->stream << class_name << "* operator->() const { return ptr; }\n";
  // operator T()
  // template <typename T>
  this->PrintIndent(this->stream);
  this->stream << "template <typename T, typename = "
               << "typename std::enable_if<std::is_convertible<UserDataRef, T>::value>::type>\n";
  this->PrintIndent(this->stream);
  this->stream << "operator T() const {return ud_ref;}\n";
  // this->PrintIndent(this->stream);
  // this->stream << "operator RTValue() const {return ud_ref;}\n";
  // this->PrintIndent(this->stream);
  // this->stream << "operator RTView() const {return ud_ref;}\n";

  this->EndScope(cls_view_scope);
  this->PrintIndent(this->stream);
  this->stream << "};\n\n";
}

void CodeGenCHost::DefineUserStructInitFunc(const ClassStmt& cls_stmt, const BaseFunc& init_func) {
  this->InitAllState();
  String reserved_keyword = "_2_71828182846";
  auto class_name = cls_stmt->name;
  auto class_view_name = FunctionNameRules::get_class_view_name(class_name);
  // __del__
  auto func_deleter_name = class_name + "_F__deleter__";
  this->PrintIndent(this->stream);
  this->stream << "void " << func_deleter_name << "(ILightUserData* ptr) { ";
  this->stream << "delete static_cast<" << class_name << "*>(ptr); }\n";
  // placement new
  auto func_pl_new_name = class_name + "_F__placement_new__";
  this->PrintIndent(this->stream);
  this->stream << "void* " << func_pl_new_name << "(void* buf) { ";
  this->stream << "return new (buf) " << class_name << "; }\n";
  // placement del
  auto func_pl_del_name = class_name + "_F__placement_del__";
  this->PrintIndent(this->stream);
  this->stream << "void " << func_pl_del_name << "(ILightUserData* ptr) { ";
  this->stream << "static_cast<" << class_name << "*>(ptr)->" << class_name << "::~" << class_name
               << "(); }\n";

  String size_of_cls_repr = "sizeof(" + class_name + ")";
  // __init__
  if (init_func.defined()) {
    auto raw_params = init_func->GetParams();
    Array<BaseExpr> params;
    params.reserve(raw_params.size());
    for (size_t i = 1; i < raw_params.size(); ++i) {
      params.push_back(raw_params[i]);
    }
    this->PrintIndent(this->stream);
    auto wrapper_func = FunctionNameRules::add_wrapper_suffix(init_func->GetGlobalName());
    this->stream << class_view_name << " " << wrapper_func << "(";
    this->PrintLineVars(
        this->stream, params, init_func->GetDefaultParams(), true, true, true, false);
    stream << ") {\n";
    auto init_scope = this->BeginScope();

    // get buffer size
    this->PrintIndent(this->stream);
    this->stream << "static auto buffer_size = UserDataRef::GetInternalBufferSize();\n";

    this->PrintIndent(this->stream);
    this->stream << "if (buffer_size < " << size_of_cls_repr << ") {\n";
    auto init_if_scope = this->BeginScope();

    {
      // new twice
      this->PrintIndent(this->stream);
      this->stream << "auto self = new " << class_name << ";\n";
      this->PrintIndent(this->stream);
      this->stream << "self->function_table" << reserved_keyword << "_ = &" << class_name
                   << "::function_table_s" << reserved_keyword << "_;\n";
      this->PrintIndent(this->stream);
      this->stream << init_func->GetGlobalName() << "(self";
      if (!params.empty()) {
        this->stream << ", ";
      }
      this->PrintLineVars(
          this->stream, params, init_func->GetDefaultParams(), false, true, false, false, true);
      this->stream << ");\n";
      this->PrintIndent(this->stream);
      this->stream << "UserDataRef self_ref(" << class_name << "::tag_s" << reserved_keyword
                   << "_, ";
      this->stream << class_name << "::var_num_s" << reserved_keyword << "_, self, "
                   << func_deleter_name << ", " << symbol::library_module_ctx << ");\n";
      // assign object node
      this->PrintIndent(this->stream);
      this->stream << "self->self_node_ptr" << reserved_keyword
                   << " = (Object*)(self_ref.get());\n";
      this->PrintIndent(this->stream);
      this->stream << "return self_ref;\n";
    }

    this->EndScope(init_if_scope);
    this->PrintIndent(this->stream);
    this->stream << "} else {\n";
    auto init_else_scope = this->BeginScope();

    {
      // placement new
      this->PrintIndent(this->stream);
      this->stream << "UserDataRef self(" << class_name << "::tag_s" << reserved_keyword << "_, ";
      this->stream << class_name << "::var_num_s" << reserved_keyword << "_, " << size_of_cls_repr
                   << ", " << func_pl_new_name << ", " << func_pl_del_name << ", "
                   << symbol::library_module_ctx << ");\n";

      this->PrintIndent(this->stream);
      this->stream << class_view_name << " self_view((" << class_name
                   << "*)self.ud_ptr_nocheck());\n";
      this->PrintIndent(this->stream);
      this->stream << "self_view->function_table" << reserved_keyword << "_ = &" << class_name
                   << "::function_table_s" << reserved_keyword << "_;\n";

      this->PrintIndent(this->stream);
      this->stream << init_func->GetGlobalName() << "(self_view";
      if (!params.empty()) {
        this->stream << ", ";
      }
      this->PrintLineVars(
          this->stream, params, init_func->GetDefaultParams(), false, true, false, false, true);
      this->stream << ");\n";
      // assign object node
      this->PrintIndent(this->stream);
      this->stream << "self_view->self_node_ptr" << reserved_keyword
                   << " = (Object*)(self.get());\n";
      this->PrintIndent(this->stream);
      this->stream << "return self;\n";
    }

    this->EndScope(init_else_scope);
    this->PrintIndent(this->stream);
    this->stream << "}\n";

    this->EndScope(init_scope);
    this->PrintIndent(this->stream);
    this->stream << "}\n\n";
  } else {
    auto init_func_name = FunctionNameRules::add_class_prefix(class_name, "__init__");
    auto wrapper_func = FunctionNameRules::add_wrapper_suffix(init_func_name);
    this->PrintIndent(this->stream);
    this->stream << class_view_name << " " << wrapper_func << "() {\n";
    auto init_scope = this->BeginScope();

    // get buffer size
    this->PrintIndent(this->stream);
    this->stream << "static auto buffer_size = UserDataRef::GetInternalBufferSize();\n";

    this->PrintIndent(this->stream);
    this->stream << "if (buffer_size < " << size_of_cls_repr << ") {\n";
    auto init_if_scope = this->BeginScope();

    {
      // new twice
      this->PrintIndent(this->stream);
      this->stream << "auto self = new " << class_name << ";\n";
      this->PrintIndent(this->stream);
      this->stream << "self->function_table" << reserved_keyword << "_ = &" << class_name
                   << "::function_table_s" << reserved_keyword << "_;\n";

      this->PrintIndent(this->stream);
      this->stream << "UserDataRef self_ref(" << class_name << "::tag_s" << reserved_keyword
                   << "_, ";
      this->stream << class_name << "::var_num_s" << reserved_keyword << "_, self, "
                   << func_deleter_name << ", " << symbol::library_module_ctx << ");\n";
      // assign object node
      this->PrintIndent(this->stream);
      this->stream << "self->self_node_ptr" << reserved_keyword
                   << " = (Object*)(self_ref.get());\n";
      this->PrintIndent(this->stream);
      this->stream << "return self_ref;\n";
    }

    this->EndScope(init_if_scope);
    this->PrintIndent(this->stream);
    this->stream << "} else {\n";
    auto init_else_scope = this->BeginScope();

    {
      // placement new
      this->PrintIndent(this->stream);
      this->stream << "UserDataRef self(" << class_name << "::tag_s" << reserved_keyword << "_, ";
      this->stream << class_name << "::var_num_s" << reserved_keyword << "_, " << size_of_cls_repr
                   << ", " << func_pl_new_name << ", " << func_pl_del_name << ", "
                   << symbol::library_module_ctx << ");\n";

      this->PrintIndent(this->stream);
      this->stream << class_view_name << " self_view((" << class_name
                   << "*)self.ud_ptr_nocheck());\n";

      this->PrintIndent(this->stream);
      this->stream << "self_view->function_table" << reserved_keyword << "_ = &" << class_name
                   << "::function_table_s" << reserved_keyword << "_;\n";

      // assign object node
      this->PrintIndent(this->stream);
      this->stream << "self_view->self_node_ptr" << reserved_keyword
                   << " = (Object*)(self.get());\n";
      this->PrintIndent(this->stream);
      this->stream << "return self;\n";
    }

    this->EndScope(init_else_scope);
    this->PrintIndent(this->stream);
    this->stream << "}\n";

    this->EndScope(init_scope);
    this->PrintIndent(this->stream);
    this->stream << "}\n\n";
  }
}

void CodeGenCHost::VisitExpr_(const ClassGetItemNode* op, std::ostream& os) {
  this->VisitExpr(op->self, os);
  os << "->" << op->attr->value;
}

void CodeGenCHost::VisitExpr_(const NoneExprNode* op, std::ostream& os) {
  os << "None";
}

void CodeGenCHost::VisitStmt_(const ExceptionHandlerNode* op, std::ostream& os) {
  MXCHECK(!op->e.defined());

  this->PrintIndent(os);
  os << "catch (...) {";
  this->PrintSpan(op->span, os);
  os << "\n";

  int body_scope = BeginScope();
  PrintStmt(op->body, os);
  this->EndScope(body_scope);

  this->PrintIndent(os);
  os << "}\n";
}

void CodeGenCHost::VisitStmt_(const TryExceptNode* op, std::ostream& os) {
  this->PrintIndent(os);
  os << "try {";
  this->PrintSpan(op->span, os);
  os << "\n";

  int body_scope = BeginScope();
  this->PrintStmt(op->body, os);
  this->EndScope(body_scope);

  this->PrintIndent(os);
  os << "}\n";

  for (auto& handler : op->handlers) {
    this->PrintStmt(handler, os);
  }
}

void CodeGenCHost::VisitStmt_(const RaiseNode* op, std::ostream& os) {
  auto py_info = this->GenPythonStyleSpanMessage(op->span, this->current_py_func_name_);
  if (op->exc.defined()) {
    auto exc_code = this->PrintExpr(op->exc);
    this->PrintIndent(os);
    os << "throw " << exc_code << ";";
    this->PrintSpan(op->span, os);
    os << "\n";
  } else {
    this->PrintIndent(os);
    os << "if (std::current_exception()) {std::rethrow_exception(std::current_exception());} ";
    os << "else {THROW_PY_RuntimeError(" << py_info << ", \"No active exception to reraise\");}";
    this->PrintSpan(op->span, os);
    os << "\n";
  }
}

void CodeGenCHost::VisitExpr_(const LambdaFunctionNode* op, std::ostream& os) {
  // clear previous generated state.
  auto func = GetRef<LambdaFunction>(op);

  // print captures
  os << "[";
  for (size_t i = 0; i < op->captures.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << this->PrintExpr(op->captures[i]);
  }
  os << "]";
  // print args and return types
  os << "(";
  PrintLineVars(os, op->params, {}, true, true, true, false, false);
  os << ") -> ";
  this->PrintType(op->ret_type, os);
  // print body
  os << " {";
  this->PrintSpan(op->span, os);
  os << "\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(op->body, os);
  this->EndScope(func_scope);
  this->PrintIndent(os);
  os << "}";
}

void CodeGenCHost::AddFunctionDeclaration(const BaseFunc& f) {
  bool is_yield_func = YieldDetector().GetYields(f).size() > 0;
  if (is_yield_func) {
    // clear previous generated state.
    this->InitFuncState(f);
    // reserve keywords
    ReserveKeywordsAsUnique();
    String func_name = f->GetGlobalName();
    String generator_name = "Generator_" + func_name;
    auto params = f->GetParams();
    if (!params.empty()) {
      this->PrintIndent(this->stream);
      stream << "template <";
      for (size_t i = 0; i < params.size(); ++i) {
        if (i != 0) {
          stream << ", ";
        }
        stream << "typename argument_type" << i;
      }
      stream << ">\n";
    }
    this->PrintIndent(this->stream);
    stream << "struct " << generator_name << ";\n";

    this->PrintIndent(this->stream);
    stream << generator_name;
    if (!params.empty()) {
      stream << "<";
      this->PrintLineVars(this->stream, params, f->GetDefaultParams(), true, false, true, false);
      stream << ">";
    }
    this->stream << " " << func_name << "_generator_raw_cc_00(";
    this->PrintLineVars(this->stream, params, f->GetDefaultParams(), false, true, true, true);
    this->stream << ");\n";
    this->stream << "int " << FunctionNameRules::add_packed_suffix(func_name)
                 << "(MATXScriptAny*, int, MATXScriptAny*, void*);\n";
  } else {
    CodeGenC::AddFunctionDeclaration(f);
  }
}

void CodeGenCHost::AddFunction(const BaseFunc& f) {
  String global_symbol = f->GetGlobalName();
  function_names_.emplace_back(global_symbol);

  YieldDetector detector;
  auto yield_stmts = detector.GetYields(f);
  if (yield_stmts.empty()) {
    if (f.as<PrimFuncNode>()) {
      CodeGenC::AddFunction(Downcast<PrimFunc>(f));
    } else {
      AssignOptimizerMutator assign_opt;
      CodeGenC::AddFunction(Downcast<Function>(assign_opt.run(f)));
      // TODO: enable move optimizer
      // MoveOptimizerMutator move_opt;
      // CodeGenC::AddFunction(Downcast<Function>(move_opt.run(f)));
    }
  } else {
    // reorder yield label id
    auto new_f = YieldLabelMutator().MutateFunc(f);
    yield_stmts = detector.GetYields(new_f);
    AddYieldFunction(new_f, yield_stmts);
  }
}

namespace {
String GenTemplateDeclare(const Array<BaseExpr>& params) {
  std::stringstream template_stream;
  // begin print generator
  if (!params.empty()) {
    template_stream << "template <";
    for (int i = 0; i < params.size(); ++i) {
      template_stream << "typename argument_type" << i;
      if (i + 1 != params.size()) {
        template_stream << ", ";
      }
    }
    template_stream << ">";
  }
  return template_stream.str();
}
String GenTemplateComposeType(const String& prefix, const Array<BaseExpr>& params) {
  std::stringstream template_stream;
  template_stream << prefix;
  // begin print generator
  if (!params.empty()) {
    template_stream << "<";
    for (int i = 0; i < params.size(); ++i) {
      template_stream << "argument_type" << i;
      if (i + 1 != params.size()) {
        template_stream << ", ";
      }
    }
    template_stream << ">";
  }
  return template_stream.str();
}
}  // namespace

void CodeGenCHost::VisitStmt_(const HLOYieldNode* op, std::ostream& os) {
  auto yield_id = Downcast<IntImm>(op->label);
  String value = PrintExpr(op->symbol);
  PrintIndent(os);
  os << "generator_state__ = " << yield_id->value << ";\n";
  PrintIndent(os);
  os << "return generator_value__ = (" << value << ");\n";
  PrintIndent(os);
  os << "yield_point" << yield_id->value << ":;\n";
}

void CodeGenCHost::AddYieldFunction(const BaseFunc& f, const std::vector<HLOYield>& yield_stmts) {
  // only support high level yield function
  Function hlo_func = Downcast<Function>(f);
  // clear previous generated state.
  this->InitFuncState(hlo_func);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<StringRef>(attr::kGlobalSymbol);
  MXCHECK(global_symbol.defined())
      << "CodeGenC: Expect Function to have the global_symbol attribute";
  auto func_name = global_symbol.value().operator String();
  String generator_name = "Generator_" + func_name;
  String yield_value_name = "generator_value__";

  // init var
  Map<BaseExpr, BaseExpr> func_var_map;
  BaseFunc f_no_alloca = SubstituteYieldFunctionVars(f, func_var_map);
  for (auto var_pair : func_var_map) {
    AllocVarID(var_pair.first);
    AllocVarID(var_pair.second);
  }

  // check yield symbol all is prim_expr ?
  bool yield_symbol_is_all_same_prim_type = true;
  runtime::DataType yield_prim_type;
  {
    bool first_set_type = true;
    for (auto& ys : yield_stmts) {
      if (!ys->symbol->IsInstance<PrimExprNode>()) {
        yield_symbol_is_all_same_prim_type = false;
        break;
      }
      auto node = ys->symbol.as<PrimExprNode>();
      if (first_set_type) {
        first_set_type = false;
        yield_prim_type = node->dtype;
      } else {
        if (yield_prim_type != node->dtype) {
          yield_symbol_is_all_same_prim_type = false;
        }
      }
    }
  }

  // parent type
  String base_generator_cls;
  {
    std::stringstream base_generator_cls_stream;
    base_generator_cls_stream << "BaseGenerator<";
    if (yield_symbol_is_all_same_prim_type) {
      this->PrintType(yield_prim_type, base_generator_cls_stream);
    } else {
      base_generator_cls_stream << "RTValue";
    }
    base_generator_cls_stream << ">";
    base_generator_cls = base_generator_cls_stream.str();
  }

  // begin print generator
  if (!hlo_func->params.empty()) {
    this->PrintIndent(this->stream);
    stream << GenTemplateDeclare(hlo_func->params) << "\n";
  }
  String generator_template = GenTemplateComposeType(generator_name, hlo_func->params);
  String yield_result_type = "typename " + generator_template + "::result_type";

  this->PrintIndent(this->stream);
  stream << "struct " << generator_name << " : " << base_generator_cls << " {\n";

  int generator_struct_scope = this->BeginScope();

  // begin typedef
  this->PrintIndent(this->stream);
  stream << "typedef GeneratorIterator<" << generator_template << "> iterator;\n";

  if (yield_symbol_is_all_same_prim_type) {
    this->PrintIndent(this->stream);
    stream << "typedef ";
    this->PrintType(yield_prim_type, stream);
    stream << " value_type;\n";
    this->PrintIndent(this->stream);
    stream << "typedef ";
    this->PrintType(yield_prim_type, stream);
    stream << " result_type;\n";
  } else {
    this->PrintIndent(this->stream);
    stream << "typedef RTValue value_type;\n";
    this->PrintIndent(this->stream);
    stream << "typedef RTValue result_type;\n";
  }

  // begin define return value
  this->PrintIndent(this->stream);
  stream << yield_result_type << " " << yield_value_name << ";\n";
  // begin define args
  std::unordered_set<const BaseExprNode*> args_var;
  if (!hlo_func->params.empty()) {
    for (int i = 0; i < hlo_func->params.size(); ++i) {
      this->PrintIndent(this->stream);
      stream << "argument_type" << i << " " << GetVarID(hlo_func->params[i]) << ";\n";
      this->PrintIndent(this->stream);
      stream << "argument_type" << i << " " << GetVarID(func_var_map[hlo_func->params[i]]) << ";\n";
      args_var.emplace(hlo_func->params[i].get());
    }
  }
  for (auto& var_pair : func_var_map) {
    if (!args_var.count(var_pair.first.get())) {
      this->PrintIndent(this->stream);
      this->PrintType(var_pair.second->checked_type(), stream);
      stream << " " << GetVarID(var_pair.second) << ";\n";
    }
  }
  // Generator constructor
  this->PrintIndent(this->stream);
  stream << generator_name << "() : " << base_generator_cls << "() {}";  // default constructor
  if (!hlo_func->params.empty()) {
    // constructor with args
    this->PrintIndent(this->stream);
    stream << generator_name << "(";
    for (int i = 0; i < hlo_func->params.size(); ++i) {
      stream << "argument_type" << i << " " << GetVarID(hlo_func->params[i]);
      if (i + 1 != hlo_func->params.size()) {
        stream << ", ";
      }
    }
    stream << ") : " << base_generator_cls << "(), ";
    for (int i = 0; i < hlo_func->params.size(); ++i) {
      stream << GetVarID(hlo_func->params[i]) << "(" << GetVarID(hlo_func->params[i]) << ")";
      if (i + 1 != hlo_func->params.size()) {
        stream << ", ";
      }
    }
    stream << " {}\n";
  }
  // iterator operator
  this->PrintIndent(this->stream);
  stream << "void operator++() { next(); }\n";
  this->PrintIndent(this->stream);
  stream << yield_result_type << " operator*() const { return " << yield_value_name << "; }\n";

  // iterator begin/end
  String iterator_name = "GeneratorIterator<" + generator_name + ">";
  this->PrintIndent(this->stream);
  stream << iterator_name << " begin() { next(); return " << iterator_name << "(*this); }\n";
  this->PrintIndent(this->stream);
  stream << iterator_name << " end() { return " << iterator_name << "(); }\n";

  // next
  this->PrintIndent(this->stream);
  stream << yield_result_type << " next();\n";

  // end struct define
  this->EndScope(generator_struct_scope);
  this->PrintIndent(this->stream);
  stream << "};\n";

  // next function declare
  if (!hlo_func->params.empty()) {
    this->PrintIndent(this->stream);
    stream << GenTemplateDeclare(hlo_func->params) << "\n";
  }
  this->PrintIndent(this->stream);
  stream << yield_result_type << " " << generator_template << "::next() {\n";
  // begin next: define switch
  int generator_struct_next_func_scope = this->BeginScope();
  this->PrintIndent(this->stream);
  stream << "switch (generator_state__) {\n";
  int generator_struct_next_func_case_scope = this->BeginScope();
  for (auto& yield_stmt : yield_stmts) {
    this->PrintIndent(this->stream);
    IntImm id = Downcast<IntImm>(yield_stmt->label);
    stream << "case " << id->value << ": goto yield_point" << id->value << ";";
  }
  this->EndScope(generator_struct_next_func_case_scope);
  this->PrintIndent(this->stream);
  stream << "};\n";

  // begin func body
  this->PrintIndent(this->stream);
  stream << "{\n";
  int generator_struct_next_func_body_scope = this->BeginScope();
  if (f_no_alloca->IsInstance<PrimFuncNode>()) {
    this->PrintStmt(Downcast<PrimFunc>(f_no_alloca)->body, this->stream);
  } else {
    this->PrintStmt(Downcast<Function>(f_no_alloca)->body, this->stream);
  }
  this->EndScope(generator_struct_next_func_body_scope);
  this->PrintIndent(this->stream);
  stream << "}\n";

  // begin final body
  this->PrintIndent(this->stream);
  stream << "{\n";
  int generator_struct_next_func_final_scope = this->BeginScope();
  this->PrintIndent(this->stream);
  stream << "generator_state__ = -1;\n";
  this->PrintIndent(this->stream);
  stream << "goto that_is_all_folks;\n";
  this->EndScope(generator_struct_next_func_final_scope);
  this->PrintIndent(this->stream);
  stream << "}\n";
  this->PrintIndent(this->stream);
  stream << "that_is_all_folks:\n";
  this->PrintIndent(this->stream);
  stream << "return result_type();\n";
  this->EndScope(generator_struct_next_func_scope);
  this->PrintIndent(this->stream);
  stream << "}\n\n";

  // print normal function
  String new_func_name = func_name + "_generator_raw_cc_00";
  this->PrintIndent(this->stream);
  stream << generator_name;
  if (!hlo_func->params.empty()) {
    stream << "<";
    this->PrintLineVars(
        this->stream, hlo_func->params, hlo_func->GetDefaultParams(), false, false, true, false);
    stream << ">";
  }
  this->stream << " " << new_func_name << "(";
  this->PrintLineVars(
      this->stream, hlo_func->params, hlo_func->GetDefaultParams(), false, true, true, false);
  this->stream << ") { return ";
  stream << generator_name;
  if (!hlo_func->params.empty()) {
    stream << "<";
    this->PrintLineVars(
        this->stream, hlo_func->params, hlo_func->GetDefaultParams(), false, false, true, false);
    stream << ">";
  }
  stream << "(";
  for (size_t i = 0; i < hlo_func->params.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << GetVarID(hlo_func->params[i]);
  }
  stream << "); }\n\n";

  // generator as ObjectRef
  const char* return_object_type = "RTValueGenerator";
  if (yield_symbol_is_all_same_prim_type) {
    if (yield_prim_type.is_bool()) {
      return_object_type = "BoolGenerator";
    } else if (yield_prim_type.is_int()) {
      MXCHECK(yield_prim_type.bits() == 32 || yield_prim_type.bits() == 64);
      if (yield_prim_type.bits() == 32) {
        return_object_type = "Int32Generator";
      } else {
        return_object_type = "Int64Generator";
      }
    } else if (yield_prim_type.is_float()) {
      MXCHECK(yield_prim_type.bits() == 32 || yield_prim_type.bits() == 64);
      if (yield_prim_type.bits() == 32) {
        return_object_type = "Float32Generator";
      } else {
        return_object_type = "Float64Generator";
      }
    } else {
      MXCHECK(false) << "yield prim type is not supported";
    }
  } else {
    return_object_type = "RTValueGenerator";
  }

  this->PrintIndent(this->stream);

  stream << return_object_type << " " << func_name << "(";
  this->PrintLineVars(this->stream, f->GetParams(), f->GetDefaultParams(), false, true, true, true);
  stream << ") {\n";
  int pack_scope_id = this->BeginScope();
  this->PrintIndent(this->stream);
  stream << base_generator_cls << "* __generator_interface__ = new " << generator_name;
  if (!hlo_func->params.empty()) {
    stream << "<";
    this->PrintLineVars(
        this->stream, hlo_func->params, hlo_func->GetDefaultParams(), false, false, true, false);
    stream << ">";
  }
  stream << "(" << new_func_name << "(";
  for (size_t i = 0; i < hlo_func->params.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << GetVarID(hlo_func->params[i]);
  }
  stream << "));\n";
  this->PrintIndent(this->stream);
  stream << "return " << return_object_type;
  stream << "(std::shared_ptr<" << base_generator_cls << ">(__generator_interface__));\n";
  this->EndScope(pack_scope_id);
  this->PrintIndent(this->stream);
  stream << "}\n";
}

void CodeGenCHost::PrintPackedFunctionMacro(const BaseFunc& f) {
  return CodeGenC::PrintPackedFunctionMacro(f);
}

void CodeGenCHost::PrintPackedFunctionMacro(const String& global_symbol,
                                            const String& bound_symbol,
                                            const Type& ret_type,
                                            const Array<BaseExpr>& args,
                                            const Array<BaseExpr>& default_args,
                                            bool first_arg_is_self,
                                            bool capture_session_handle,
                                            const Span& span) {
  return CodeGenC::PrintPackedFunctionMacro(global_symbol,
                                            bound_symbol,
                                            ret_type,
                                            args,
                                            default_args,
                                            first_arg_is_self,
                                            capture_session_handle,
                                            span);
}

void CodeGenCHost::PrintFuncPrefix(ir::Type ret_type) {  // NOLINT(*)
  stream << "MATX_DLL ";
  PrintType(ret_type, stream);
}

void CodeGenCHost::PrintFinalReturn() {  // NOLINT(*)
  // this->PrintIndent(this->stream);
  // stream << "return 0;\n";
}

void CodeGenCHost::PrintType(const ir::Type& t, std::ostream& os) {  // NOLINT(*)
  if (auto node = t.as<ClassTypeNode>()) {
    auto self_tn = FunctionNameRules::get_class_view_name(node->header->name_hint.view());
    os << self_tn;
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenCHost::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    MXCHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  MXLOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenCHost::VisitExpr_(const PrimCallNode* op, std::ostream& os) {  // NOLINT(*)
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenCHost::VisitExpr_(const PrimMinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::VisitExpr_(const PrimMaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

static Stmt BuildForStmtFromComprehension(const ComprehensionNode* op, Stmt body) {
  static PrimType int64_type(runtime::DataType::Int(64));

  // process body
  if (op->ifs.size()) {
    for (auto predicate : op->ifs) {
      body = IfThenElse(predicate, body, Stmt{nullptr}, body->span);
    }
  }

  Array<Stmt> seq_stmt;
  if (const auto* range_node = op->iter.as<RangeExprNode>()) {
    auto fn_eval_iter = [&](BaseExpr expr, const char* prefix) -> BaseExpr {
      if (!(expr->IsInstance<PrimVarNode>() || expr->IsInstance<IntImmNode>())) {
        if (!IsPrimType(expr)) {
          expr = HLOCastPrim(runtime::DataType::Int(64), expr, body->span);
        }
        // TODO: fixname
        String new_name(prefix);
        AllocaVarStmt alloca_stmt(new_name, int64_type, expr, body->span);
        seq_stmt.push_back(alloca_stmt);
        expr = alloca_stmt->var;
      }
      return expr;
    };
    BaseExpr start = fn_eval_iter(range_node->start, "start");
    BaseExpr stop = fn_eval_iter(range_node->stop, "stop");
    BaseExpr step = fn_eval_iter(range_node->step, "step");
    MXCHECK(op->target->IsInstance<PrimVarNode>()) << "internal error";
    For loop_stmt(Downcast<PrimVar>(op->target),
                  std::move(start),
                  std::move(stop),
                  std::move(step),
                  ForType::Serial,
                  body,
                  body->span);
    seq_stmt.push_back(loop_stmt);
  } else {
    BaseExpr iter = op->iter;
    if (!(iter->IsInstance<HLOVarNode>())) {
      AllocaVarStmt alloca_stmt("iter", iter->checked_type_, iter, body->span);
      seq_stmt.push_back(alloca_stmt);
      iter = alloca_stmt->var;
    }
    if (auto const* tup = op->target.as<TupleExprNode>()) {
      AutoFor loop_stmt(tup->fields, std::move(iter), body, body->span);
      seq_stmt.push_back(loop_stmt);
    } else {
      AutoFor loop_stmt(op->target, std::move(iter), body, body->span);
      seq_stmt.push_back(loop_stmt);
    }
  }
  if (seq_stmt.size() == 1) {
    return seq_stmt[0];
  }
  return SeqStmt(seq_stmt, body->span);
}

void CodeGenCHost::VisitExpr_(const ListCompNode* op, std::ostream& os) {
  // capture all free_var
  os << "[&]";
  // print args and return types
  os << "() -> ";
  this->PrintType(op->checked_type_, os);
  // print body begin
  os << " {";
  this->PrintSpan(op->span, os);
  os << "\n";
  int func_scope = this->BeginScope();

  // make body
  Array<Stmt> body;
  {
    auto const* li_ty_node = op->checked_type_.as<ListTypeNode>();
    MXCHECK(li_ty_node) << "internal error";
    AllocaVarStmt alloc_stmt(
        "__reserved_list_comp_result", op->checked_type_, BaseExpr{nullptr}, op->span);
    body.push_back(alloc_stmt);
    {
      HLOExpr call_op =
          li_ty_node->is_full_typed ? builtin::ft_list_reserve() : builtin::list_reserve();
      Call call_list_reserve(VoidType(), call_op, {alloc_stmt->var}, op->span, {});
      // TODO: fix reserve
      // body.push_back(ExprStmt(call_list_reserve, op->span));
    }
    Stmt last_stmt{nullptr};
    {
      HLOExpr call_op =
          li_ty_node->is_full_typed ? builtin::ft_list_append() : builtin::list_append();
      Call call_list_append(VoidType(), call_op, {alloc_stmt->var, op->elt}, op->span, {});
      last_stmt = ExprStmt(call_list_append, op->span);
    }

    for (auto gen_itr = op->generators.rbegin(); gen_itr != op->generators.rend(); ++gen_itr) {
      last_stmt = BuildForStmtFromComprehension((*gen_itr).get(), last_stmt);
    }
    body.push_back(last_stmt);
    // return
    body.push_back(ReturnStmt(alloc_stmt->var, op->span));
  }

  // visit body
  this->PrintStmt(SeqStmt(body, op->span), os);

  // print body end
  this->EndScope(func_scope);
  this->PrintIndent(os);
  os << "}";
  // print call
  os << "()";
}

void CodeGenCHost::VisitExpr_(const SetCompNode* op, std::ostream& os) {
  // capture all free_var
  os << "[&]";
  // print args and return types
  os << "() -> ";
  this->PrintType(op->checked_type_, os);
  // print body begin
  os << " {";
  this->PrintSpan(op->span, os);
  os << "\n";
  int func_scope = this->BeginScope();

  // make body
  Array<Stmt> body;
  {
    auto const* set_ty_node = op->checked_type_.as<SetTypeNode>();
    MXCHECK(set_ty_node) << "internal error";
    AllocaVarStmt alloc_stmt(
        "__reserved_set_comp_result", op->checked_type_, BaseExpr{nullptr}, op->span);
    body.push_back(alloc_stmt);
    {
      HLOExpr call_op =
          set_ty_node->is_full_typed ? builtin::ft_set_reserve() : builtin::set_reserve();
      Call call_set_reserve(VoidType(), call_op, {alloc_stmt->var}, op->span, {});
      // TODO: fix reserve
      // body.push_back(ExprStmt(call_set_reserve, op->span));
    }
    Stmt last_stmt{nullptr};
    {
      HLOExpr call_op = set_ty_node->is_full_typed ? builtin::ft_set_add() : builtin::set_add();
      Call call_set_add(VoidType(), call_op, {alloc_stmt->var, op->elt}, op->span, {});
      last_stmt = ExprStmt(call_set_add, op->span);
    }

    for (auto gen_itr = op->generators.rbegin(); gen_itr != op->generators.rend(); ++gen_itr) {
      last_stmt = BuildForStmtFromComprehension((*gen_itr).get(), last_stmt);
    }
    body.push_back(last_stmt);
    // return
    body.push_back(ReturnStmt(alloc_stmt->var, op->span));
  }

  // visit body
  this->PrintStmt(SeqStmt(body, op->span), os);

  // print body end
  this->EndScope(func_scope);
  this->PrintIndent(os);
  os << "}";
  // print call
  os << "()";
}

void CodeGenCHost::VisitExpr_(const DictCompNode* op, std::ostream& os) {
  // capture all free_var
  os << "[&]";
  // print args and return types
  os << "() -> ";
  this->PrintType(op->checked_type_, os);
  // print body begin
  os << " {";
  this->PrintSpan(op->span, os);
  os << "\n";
  int func_scope = this->BeginScope();

  // make body
  Array<Stmt> body;
  {
    auto const* dict_ty_node = op->checked_type_.as<DictTypeNode>();
    MXCHECK(dict_ty_node) << "internal error";
    AllocaVarStmt alloc_stmt(
        "__reserved_dict_comp_result", op->checked_type_, BaseExpr{nullptr}, op->span);
    body.push_back(alloc_stmt);
    {
      HLOExpr call_op =
          dict_ty_node->is_full_typed ? builtin::ft_dict_reserve() : builtin::dict_reserve();
      Call call_dict_reserve(VoidType(), call_op, {alloc_stmt->var}, op->span, {});
      // TODO: fix reserve
      // body.push_back(ExprStmt(call_dict_reserve, op->span));
    }
    Stmt last_stmt{nullptr};
    {
      HLOExpr call_op = dict_ty_node->is_full_typed ? builtin::ft_dict___setitem__()
                                                    : builtin::dict___setitem__();
      Call call_dict_setitem(
          VoidType(), call_op, {alloc_stmt->var, op->key, op->value}, op->span, {});
      last_stmt = ExprStmt(call_dict_setitem, op->span);
    }

    for (auto gen_itr = op->generators.rbegin(); gen_itr != op->generators.rend(); ++gen_itr) {
      last_stmt = BuildForStmtFromComprehension((*gen_itr).get(), last_stmt);
    }
    body.push_back(last_stmt);
    // return
    body.push_back(ReturnStmt(alloc_stmt->var, op->span));
  }

  // visit body
  this->PrintStmt(SeqStmt(body, op->span), os);

  // print body end
  this->EndScope(func_scope);
  this->PrintIndent(os);
  os << "}";
  // print call
  os << "()";
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T* op,
                                               const char* compare,
                                               std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  String a_id = SSAGetID(temp_a.str(), op->a.dtype(), os);
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  String b_id = SSAGetID(temp_b.str(), op->b.dtype(), os);

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

void CodeGenCHost::GenerateFuncRegistry(const std::vector<String>& func_names,
                                        const String& class_name) {
  stream << "extern \"C\" {\n\n";
  stream << "MATX_DLL MATXScriptBackendPackedCFunc " << symbol::library_func_array << class_name
         << "[] = {\n";
  for (auto& f : func_names) {
    stream << "    (MATXScriptBackendPackedCFunc)" << FunctionNameRules::add_packed_suffix(f)
           << ",\n";
  }
  stream << "};\n";
  auto registry = GenerateFuncRegistryNames(func_names);
  stream << "MATX_DLL MATXScriptFuncRegistry " << symbol::library_func_registry << class_name
         << " = {\n"
         << "    \"" << runtime::BytesEscape(registry.data(), registry.size(), true) << "\","
         << "    " << symbol::library_func_array << class_name << ",\n"
         << "};\n";
  stream << "\n} // extern C\n\n";
}

void CodeGenCHost::GenerateClosuresNames(const std::vector<String>& func_names) {
  stream << "extern \"C\" {\n\n";
  stream << "MATX_DLL const char* " << symbol::library_closures_names << " = ";
  auto registry = GenerateFuncRegistryNames(func_names);
  stream << "\"" << runtime::BytesEscape(registry.data(), registry.size(), true) << "\";\n";
  stream << "\n} // extern C\n\n";
}

void CodeGenCHost::GenerateCrtSystemLib() {
  stream << "static const MATXModule _matx_system_lib = {\n"
         << "    &_matx_func_registry,\n"
         << "};\n"
         << "const MATXModule* MATXSystemLibEntryPoint(void) {\n"
         << "    return &_matx_system_lib;\n"
         << "}\n";
}

StringRef BuildPrimFuncCHost(PrimFunc f, StringRef func_name = "__main__") {
  auto func = WithAttr(std::move(f), attr::kGlobalSymbol, func_name);
  CodeGenCHost cg;
  cg.AddFunction(func);
  String code = cg.Finish();
  return code;
}

MATXSCRIPT_REGISTER_GLOBAL("codegen.build.c").set_body_typed(BuildPrimFuncCHost);

static void TypeVisitFunc(ClassType input,
                          std::vector<ClassType>& outputs,
                          std::unordered_set<StringRef>& visited,
                          const std::unordered_map<StringRef, ClassType>& defines) {
  for (auto& var_t : input->var_types) {
    if (auto* node = var_t.as<ClassTypeNode>()) {
      MXCHECK(defines.count(node->header->name_hint));
      TypeVisitFunc(defines.at(node->header->name_hint), outputs, visited, defines);
    }
  }
  if (auto* node = input->base.as<ClassTypeNode>()) {
    MXCHECK(defines.count(node->header->name_hint));
    TypeVisitFunc(defines.at(node->header->name_hint), outputs, visited, defines);
  }
  if (!visited.count(input->header->name_hint)) {
    visited.emplace(input->header->name_hint);
    outputs.push_back(std::move(input));
  }
}

static Function GetUnboundFunction(const Function& f) {
  if (!f->IsClassMember()) {
    return Function{};
  }
  auto params = f->GetParams();
  if (params.empty()) {
    return Function{};
  }
  auto self_node = params[0].as<HLOVarNode>();
  MXCHECK(self_node != nullptr);
  auto new_self_node = runtime::make_object<HLOVarNode>(*self_node);
  new_self_node->vid = Id("self");
  HLOVar self(new_self_node);
  params.Set(0, self);
  Array<BaseExpr> pass_args;
  for (auto i = 1; i < params.size(); ++i) {
    pass_args.push_back(params[i]);
  }
  auto attrs = runtime::make_object<DictAttrsNode>(*f->attrs.get());
  attrs->dict.erase(attr::kClassNameBelongTo);
  ReturnStmt body(Call(f->GetReturnType(),
                       ClassGetItem(self, StringImm(f->GetBoundName())),
                       pass_args,
                       f->span,
                       Downcast<Array<ObjectRef>>(f->type_params)),
                  f->span);
  return Function(
      params, f->default_params, body, f->ret_type, f->type_params, DictAttrs(attrs), f->span);
}

static BaseFunc RunOptimizations(BaseFunc func) {
  // Optimizer
  FuncArgsOptimizerMutator args_opt;
  FuseContBinaryAddOptimizer fuse_cont_bin_add_opt;
  FuseContAnyGetSetItemOptimizer fuse_cont_get_set_item_opt;
  FuseContCasterOptimizer fuse_cont_caster_opt;

  func = fuse_cont_get_set_item_opt.run(func);
  func = fuse_cont_caster_opt.run(func);
  bool is_yield_func = YieldDetector().GetYields(func).size() > 0;
  if (!is_yield_func) {
    func = args_opt.run(func);
  }
  return func;
}

runtime::Module BuildCHost(IRModule mod) {
  using ::matxscript::runtime::FunctionRegistry;

  // TODO: clean code
  Array<BaseFunc> mod_functions;
  Array<ClassStmt> mod_classes;
  for (auto stmt : mod->body) {
    if (const auto* cls_node = stmt.as<ClassStmtNode>()) {
      auto cls_stmt = runtime::GetRef<ClassStmt>(cls_node);
      Array<Stmt> cls_new_body;
      for (auto cls_stmt : cls_node->body) {
        MXCHECK(cls_stmt->IsInstance<BaseFuncNode>()) << "internal error";
        cls_new_body.push_back(RunOptimizations(Downcast<BaseFunc>(cls_stmt)));
      }
      cls_stmt.CopyOnWrite()->body = cls_new_body;
      mod_classes.push_back(cls_stmt);
    } else if (const auto* fn_node = stmt.as<BaseFuncNode>()) {
      auto func = RunOptimizations(GetRef<BaseFunc>(fn_node));
      mod_functions.push_back(func);
    } else {
      MXTHROW << "[BuildCHost] unsupported stmt: " << stmt;
    }
  }

  // Find class init function
  auto FindInitFunc = [](const ClassStmt& cls) -> BaseFunc {
    for (auto stmt : cls->body) {
      if (auto* fn_node = stmt.as<BaseFuncNode>()) {
        if (fn_node->IsClassConstructor()) {
          return GetRef<BaseFunc>(fn_node);
        }
      }
    }
    return BaseFunc(nullptr);
  };

  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts);

  for (auto& cls : mod_classes) {
    cg.InitTypeRegistry(cls);
  }

  cg.BeginAnonymousNamespace();

  // class methods
  std::unordered_map<const void*, std::unordered_map<String, BaseFunc>> class_methods;
  // Add User Data declaration
  for (auto& cls : mod_classes) {
    for (auto stmt : cls->body) {
      BaseFunc fn = Downcast<BaseFunc>(stmt);
      class_methods[cls.get()][fn->GetGlobalName()] = fn;
    }
  }

  // Add User Data declaration
  for (auto& cls : mod_classes) {
    cg.AddUserStructDeclaration(cls);
  }

  // Add User Data init wrapper function declaration
  for (auto& cls : mod_classes) {
    auto init_func = FindInitFunc(cls);
    cg.AddUserStructInitDeclaration(cls, init_func);
  }

  // Add Function forward declaration
  for (auto fn : mod_functions) {
    if (fn->IsInstance<PrimFuncNode>()) {
      auto prim_fn = Downcast<PrimFunc>(fn);
      cg.AddFunctionDeclaration(prim_fn);
    } else {
      auto hlo_fn = Downcast<Function>(fn);
      cg.AddFunctionDeclaration(hlo_fn);
    }
  }
  for (auto cls : mod_classes) {
    for (auto fn : cls->body) {
      if (fn->IsInstance<PrimFuncNode>()) {
        auto prim_fn = Downcast<PrimFunc>(fn);
        cg.AddFunctionDeclaration(prim_fn);
      } else {
        auto hlo_fn = GetUnboundFunction(Downcast<Function>(fn));
        cg.AddFunctionDeclaration(hlo_fn);
      }
    }
  }

  std::vector<String> func_names;
  std::vector<String> closures_names;
  std::unordered_map<StringRef, std::vector<String>> class_func_names;

  // Add User Data define
  for (auto& cls : mod_classes) {
    class_func_names.emplace(cls->name, std::vector<String>());
    auto init_func = FindInitFunc(cls);
    if (init_func.defined()) {
      auto wrapper_func = FunctionNameRules::add_wrapper_suffix(init_func->GetGlobalName());
      func_names.push_back(wrapper_func);
      closures_names.push_back(wrapper_func);
      class_func_names[cls->name].emplace_back(wrapper_func);
    }
    cg.DefineUserStruct(cls, class_methods[cls.get()]);
    for (auto& member_fn : cls->type->unbound_func_names) {
      class_func_names[cls->name].emplace_back(member_fn.operator String());
    }
  }

  // Add User Data init wrapper function define
  for (auto& cls : mod_classes) {
    auto init_func = FindInitFunc(cls);
    cg.DefineUserStructInitFunc(cls, init_func);
  }

  for (auto fn : mod_functions) {
    cg.AddFunction(fn);
    cg.PrintPackedFunctionMacro(fn);
    if (fn->CaptureSessionHandle()) {
      closures_names.push_back(fn->GetGlobalName());
    }
    if (fn->ExportSymbol()) {
      func_names.push_back(fn->GetGlobalName());
    }
  }

  for (auto cls : mod_classes) {
    for (auto stmt : cls->body) {
      auto fn = Downcast<BaseFunc>(stmt);
      cg.AddFunction(fn);

      auto f = GetUnboundFunction(Downcast<Function>(fn));
      cg.AddFunction(f);
      cg.PrintPackedFunctionMacro(f);

      if (fn->CaptureSessionHandle()) {
        closures_names.push_back(fn->GetGlobalName());
      }
      if (fn->ExportSymbol()) {
        func_names.push_back(fn->GetGlobalName());
      }
      if (fn->IsClassConstructor()) {
        auto wrapper_func = FunctionNameRules::add_wrapper_suffix(fn->GetGlobalName());
        auto raw_params = fn->GetParams();
        MXCHECK((!raw_params.empty())) << "__init__ function has no self arg ???";
        auto new_params = Array<BaseExpr>();
        for (auto i = 1; i < raw_params.size(); ++i) {
          new_params.push_back(raw_params[i]);
        }
        cg.PrintPackedFunctionMacro(wrapper_func,
                                    fn->GetBoundName(),
                                    raw_params[0]->checked_type(),
                                    new_params,
                                    fn->GetDefaultParams(),
                                    false,
                                    fn->CaptureSessionHandle(),
                                    fn->span);
      }
    }
  }

  cg.EndAnonymousNamespace();

  for (auto& cls_mem : class_func_names) {
    cg.GenerateFuncRegistry(cls_mem.second, cls_mem.first);
  }

  cg.GenerateFuncRegistry(func_names);
  cg.GenerateClosuresNames(closures_names);
  // cg.GenerateCrtSystemLib();

  String code = cg.Finish();
  return CSourceModuleCreate(code, "c");
}

runtime::Module BuildEembeddedCHost(String code) {
  return CSourceModuleCreate(code, "c");
}

MATXSCRIPT_REGISTER_GLOBAL("module.build.c").set_body_typed(BuildCHost);
MATXSCRIPT_REGISTER_GLOBAL("embedded.build.c").set_body_typed(BuildEembeddedCHost);

}  // namespace codegen
}  // namespace matxscript
