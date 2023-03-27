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

class IRModule;
/*!
 * \brief IRModule that holds functions and classes.
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
  /*! \brief the functions, classes and so on. */
  Array<Stmt> body;

  IRModuleNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("body", &body);
  }

  MATX_DLL bool SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const;

  MATX_DLL void SHashReduce(SHashReducer hash_reduce) const;

  /*!
   * \brief Change a function to public.
   * \param stmt The function, class or others.
   */
  MATX_DLL void AddExportFunction(const StringRef& func_name);

  /*!
   * \brief Add a stmt to the module.
   * \param stmt The function, class or others.
   */
  MATX_DLL void Add(const Stmt& stmt);

  /*!
   * \brief Update the stmts inside this environment by
   *        stmts in another environment.
   * \param other The other environment.
   */
  MATX_DLL void Update(const IRModule& other);

  static constexpr const char* _type_key = "ir.IRModule";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IRModuleNode, Object);

 private:
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
   * \param body Stmts in the module.
   */
  MATX_DLL explicit IRModule(Array<Stmt> body);

  /*! \brief default constructor */
  IRModule() : IRModule(Array<Stmt>({})) {
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
