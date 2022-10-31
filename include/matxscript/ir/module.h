// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
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
 * \file matx/ir/module.h
 * \brief IRModule that holds the functions and type definitions.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <matxscript/ir/adt.h>
#include <matxscript/ir/expr.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace ir {
// TODO(maxiandi) : remove relay parser
class IRModule;
/*!
 * \brief IRModule that holds functions and type definitions.
 *
 *  IRModule is the basic unit for all IR transformations across the stack.
 *
 *  Many operations require access to the global IRModule.
 *  We pass the IRModule by value in a functional style as an explicit argument,
 *  but we mutate the Module while optimizing programs.
 * \sa IRModule
 */
class IRModuleNode : public Object {
 public:
  /*! \brief A map from ids to all global functions. */
  runtime::Map<GlobalVar, BaseFunc> functions;
  /*! \brief A map from global type vars to User type data. */
  runtime::Map<GlobalTypeVar, ClassType> type_definitions;

  IRModuleNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("functions", &functions);
    v->Visit("type_definitions", &type_definitions);
    v->Visit("global_var_map_", &global_var_map_);
    v->Visit("global_type_var_map_", &global_type_var_map_);
  }

  MATX_DLL bool SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const;

  MATX_DLL void SHashReduce(SHashReducer hash_reduce) const;

  MATX_DLL void AddExportFunction(const StringRef& func_name);

  /*!
   * \brief Add a function to the global environment.
   * \param var The var of the global function.
   * \param func The function.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  MATX_DLL void Add(const GlobalVar& var, const BaseFunc& func, bool update = false);

  /*!
   * \brief Add a function to the global environment.
   * \param var The name of the global function.
   * \param func The function.
   *
   * It does not do type inference as Add does.
   */
  MATX_DLL void AddUnchecked(const GlobalVar& var, const BaseFunc& func);

  /*!
   * \brief Add a type-level definition to the global environment.
   * \param var The var of the global type definition.
   * \param type The User ClassType.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  MATX_DLL void AddTypeDef(const GlobalTypeVar& var, const ClassType& type, bool update = false);

  /*!
   * \brief Add a type-level definition to the global environment.
   * \param var The var of the global type definition.
   * \param type The User ClassType.
   * \param update Controls whether you can replace a definition in the
   * environment.
   *
   * It does not do type checking as AddTypeDef does.
   */
  MATX_DLL void AddTypeDefUnchecked(const GlobalTypeVar& var,
                                    const ClassType& type,
                                    bool update = false);

  /*!
   * \brief Update a function in the global environment.
   * \param var The name of the global function to update.
   * \param func The new function.
   */
  MATX_DLL void Update(const GlobalVar& var, const BaseFunc& func);

  /*!
   * \brief Update a type definition in the global environment.
   * \param var The name of the global type definition to update.
   * \param type The User ClassType.
   */
  MATX_DLL void UpdateTypeDef(const GlobalTypeVar& var, const ClassType& type);

  /*!
   * \brief Remove a function from the global environment.
   * \param var The name of the global function to update.
   */
  MATX_DLL void Remove(const GlobalVar& var);

  /*!
   * \brief Check if the global_var_map_ contains a global variable.
   * \param name The variable name.
   * \returns true if contains, otherise false.
   */
  MATX_DLL bool ContainGlobalVar(const StringRef& name) const;

  /*!
   * \brief Check if the global_type_var_map_ contains a global type variable.
   * \param name The variable name.
   * \returns true if contains, otherise false.
   */
  MATX_DLL bool ContainGlobalTypeVar(const StringRef& name) const;

  /*!
   * \brief Lookup a global function by its variable.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  MATX_DLL GlobalVar GetGlobalVar(const StringRef& str) const;

  /*!
   * \brief Collect all global vars defined in this module.
   * \returns An array of global vars
   */
  MATX_DLL runtime::Array<GlobalVar> GetGlobalVars() const;

  /*!
   * \brief Look up a global function by its name.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  MATX_DLL GlobalTypeVar GetGlobalTypeVar(const StringRef& str) const;

  /*!
   * \brief Collect all global type vars defined in this module.
   * \returns An array of global type vars
   */
  MATX_DLL runtime::Array<GlobalTypeVar> GetGlobalTypeVars() const;

  /*!
   * \brief Look up a global function by its variable.
   * \param var The global var to lookup.
   * \returns The function named by the variable argument.
   */
  MATX_DLL BaseFunc Lookup(const GlobalVar& var) const;

  /*!
   * \brief Look up a global function by its string name
   * \param name The name of the function.
   * \returns The function named by the argument.
   */
  MATX_DLL BaseFunc Lookup(const StringRef& name) const;

  /*!
   * \brief Look up a global type definition by its variable.
   * \param var The var of the global type definition.
   * \return The type definition.
   */
  MATX_DLL ClassType LookupTypeDef(const GlobalTypeVar& var) const;

  /*!
   * \brief Look up a global type definition by its name.
   * \param var The name of the global type definition.
   * \return The type definition.
   */
  MATX_DLL ClassType LookupTypeDef(const StringRef& var) const;

  /*!
   * \brief Update the functions inside this environment by
   *        functions in another environment.
   * \param other The other environment.
   */
  MATX_DLL void Update(const IRModule& other);

  /*!
   * \brief The set of imported files.
   */
  MATX_DLL std::unordered_set<StringRef> Imports() const;

  static constexpr const char* _type_key = "IRModule";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IRModuleNode, Object);

 private:
  /*! \brief A map from string names to global variables that
   * ensures global uniqueness.
   */
  runtime::Map<StringRef, GlobalVar> global_var_map_;

  /*! \brief A map from string names to global type variables (ADT names)
   * that ensures global uniqueness.
   */
  runtime::Map<StringRef, GlobalTypeVar> global_type_var_map_;

  /*! \brief The files previously imported, required to ensure
      importing is idempotent for each module.
   */
  std::unordered_set<StringRef> import_set_;
  friend class IRModule;
};

/*!
 * \brief Managed reference class to IRModuleNode.
 * \sa IRModuleNode
 */
class IRModule : public ObjectRef {
 public:
  /*!
   * \brief constructor
   * \param functions Functions in the module.
   * \param type_definitions Type definitions in the module.
   * \param import_set Set of imported files in the module
   * \param map The module source map.
   */
  MATX_DLL explicit IRModule(
      runtime::Map<GlobalVar, BaseFunc> functions,
      runtime::Map<GlobalTypeVar, ClassType> type_definitions = {},
      std::unordered_set<StringRef> import_set = std::unordered_set<StringRef>{});

  /*! \brief default constructor */
  IRModule() : IRModule(runtime::Map<GlobalVar, BaseFunc>({})) {
  }
  /*!
   * \brief constructor
   * \param n The object pointer.
   */
  explicit IRModule(ObjectPtr<Object> n) : ObjectRef(n) {
  }
  /*! \return mutable pointers to the node. */
  IRModuleNode* operator->() const {
    auto* ptr = get_mutable();
    MXCHECK(ptr != nullptr);
    return static_cast<IRModuleNode*>(ptr);
  }

  /*!
   * \brief Construct a module from a standalone expression.
   *
   * Allows one to optionally pass a global function map and
   * map of type definitions as well.
   *
   * \param expr The expression to set as the main function to the module.
   * \param global_funcs The global function map.
   * \param type_definitions Map of global type definitions
   *
   * \returns A module with expr set as the main function.
   */
  MATX_DLL static IRModule FromExpr(
      const HLOExpr& expr,
      const runtime::Map<GlobalVar, BaseFunc>& global_funcs = {},
      const runtime::Map<GlobalTypeVar, ClassType>& type_definitions = {});

  /*! \brief Declare the container type. */
  using ContainerType = IRModuleNode;

  /*! \brief Declare whether Ref is nullable. */
  static constexpr bool _type_is_nullable = false;

  // allow copy on write.
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(IRModuleNode);
};

/*!
 * \brief Pretty print a node for debug purposes.
 *
 * \param node The node to be printed.
 * \return The text reperesentation.
 * \note This function does not show version or meta-data.
 *       Use AsText if you want to store the text.
 * \sa AsText.
 */
MATX_DLL StringRef PrettyPrint(const ObjectRef& node);

/*!
 * \brief Render the node as a string in the text format.
 *
 * \param node The node to be rendered.
 * \param show_meta_data Whether to print meta data section.
 * \param annotate An optional callback function for attaching
 *        additional comment block to an expr.
 *
 * \note We support a limited set of IR nodes that are part of
 *       relay IR and
 *
 * \sa PrettyPrint.
 * \return The text representation.
 */
MATX_DLL StringRef AsText(const ObjectRef& node,
                          bool show_meta_data = true,
                          runtime::TypedNativeFunction<StringRef(ObjectRef)> annotate = nullptr);
}  // namespace ir
}  // namespace matxscript
