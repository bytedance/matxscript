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

/*!
 * \file matx/ir/class_stmt.h
 * \brief Class nodes.
 */
#pragma once

#include <string>
#include <type_traits>

#include <matxscript/ir/_base/optional_ref.h>
#include <matxscript/ir/adt.h>
#include <matxscript/ir/attrs.h>
#include <matxscript/ir/stmt.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Class that contains Function statements.
 *
 * \sa ClassStmt
 */
class ClassStmtNode : public StmtNode {
 public:
  StringRef name;
  /*! \brief the base class and it's type is always ClassStmt */
  Stmt base;
  /*! \brief methods and so on*/
  Array<Stmt> body;
  /*! \brief class type info. */
  ClassType type;

  /*! \brief Additional attributes storing the meta-data */
  DictAttrs attrs;

  /*!
   * \brief Get a class attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TObjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const ClassStmt& f) {
   *    auto value = f->GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const StringRef& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    if (!attrs.defined())
      return default_value;
    auto it = attrs->dict.find(attr_key);
    if (it != attrs->dict.end()) {
      return runtime::Downcast<Optional<TObjectRef>>((*it).second);
    } else {
      return default_value;
    }
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const StringRef& attr_key, TObjectRef&& default_value) const {
    return GetAttr<TObjectRef>(attr_key,
                               Optional<TObjectRef>(std::forward<TObjectRef>(default_value)));
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("base", &base);
    v->Visit("body", &body);
    v->Visit("type", &type);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ClassStmtNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(base, other->base) && equal(body, other->body) &&
           equal(type, other->type) && equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(name);
    hash_reduce.DefHash(base);
    hash_reduce(body);
    hash_reduce(type);
    hash_reduce(attrs);
  }

  static constexpr const char* _type_key = "ir.ClassStmt";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ClassStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to ClassStmtNode.
 * \sa ClassStmtNode
 */
class ClassStmt : public Stmt {
 public:
  MATX_DLL explicit ClassStmt(StringRef name,
                              Stmt base,
                              Array<Stmt> body,
                              ClassType cls_type,
                              DictAttrs attrs = NullValue<DictAttrs>(),
                              Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ClassStmt, Stmt, ClassStmtNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ClassStmtNode);
};

/*!
 * \brief Create a new ClassStmt that copies ClassStmt, but overrides
 *        the attribute value key with the value.
 *
 * \param stmt The input ClassStmt.
 * \param attr_key The attribute key.
 * \param attr_value The value attribute value.
 *
 * \returns The new ClassStmt with updated attributes.
 *
 * \note This function performs copy on write optimization for ClassStmt.
 *       If we move a uniquely referenced func into WithAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  cls_stmt = WithAttr(std::move(cls_stmt), "key1", value1);
 *  cls_stmt = WithAttr(std::move(cls_stmt), "key2", value2);
 *
 * \endcode
 */
inline ClassStmt WithAttr(ClassStmt stmt, StringRef attr_key, ObjectRef attr_value) {
  using TNode = typename ClassStmt::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = stmt.CopyOnWrite();
  if (node->attrs.defined()) {
    node->attrs.CopyOnWrite()->dict.Set(attr_key, attr_value);
  } else {
    Map<StringRef, ObjectRef> dict = {{attr_key, attr_value}};
    node->attrs = DictAttrs(dict);
  }
  return stmt;
}

}  // namespace ir
}  // namespace matxscript
