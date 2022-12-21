// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/apache/tvm/blob/v0.7/include/tvm/ir/attrs.h
 * with fixes applied:
 * - add namespace matxscript::ir for fix conflict with tvm
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
 * \file matx/ir/attrs.h
 * \brief Helpers for attribute objects.
 *
 *  This module enables declaration of named attributes
 *  which support default value setup and bound checking.
 *
 * \code
 *   struct MyAttrs : public matxscript::ir::AttrsNode<MyAttrs> {
 *     float learning_rate;
 *     int num_hidden;
 *     String name;
 *     // declare attribute fields in header file
 *     MATXSCRIPT_DECLARE_ATTRS(MyAttrs, "attrs.MyAttrs") {
 *       MATXSCRIPT_ATTR_FIELD(num_hidden).set_lower_bound(1);
 *       MATXSCRIPT_ATTR_FIELD(learning_rate).set_default(0.01f);
 *       MATXSCRIPT_ATTR_FIELD(name).set_default("hello");
 *     }
 *   };
 *   // register it in cc file
 *   MATXSCRIPT_REGISTER_NODE_TYPE(MyAttrs);
 * \endcode
 *
 * \sa AttrsNode, MATXSCRIPT_DECLARE_ATTRS, MATXSCRIPT_ATTR_FIELD
 */
#pragma once

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/ir/_base/structural_hash.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/prim_expr.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Create a NodeRef type that represents null.
 * \tparam TNodeRef the type to be created.
 * \return A instance that will represent None.
 */
template <typename TObjectRef>
inline TObjectRef NullValue() {
  static_assert(TObjectRef::_type_is_nullable, "Can only get NullValue for nullable types");
  return TObjectRef(ObjectPtr<Object>(nullptr));
}

template <>
inline runtime::DataType NullValue<runtime::DataType>() {
  return runtime::DataType(runtime::DataType::kHandle, 0, 0);
}

/*! \brief Error thrown during attribute checking. */
struct AttrError : public std::runtime_error {
  /*!
   * \brief constructor
   * \param msg error message
   */
  explicit AttrError(std::string msg) : std::runtime_error("AttributeError:" + msg) {
  }
};

/*!
 * \brief Information about attribute fields in string representations.
 */
class AttrFieldInfoNode : public Object {
 public:
  /*! \brief name of the field */
  StringRef name;
  /*! \brief type docstring information in str. */
  StringRef type_info;
  /*! \brief detailed description of the type */
  StringRef description;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("type_info", &type_info);
    v->Visit("description", &description);
  }

  static constexpr const char* _type_key = "AttrFieldInfo";
  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AttrFieldInfoNode, Object);
};

/*! \brief AttrFieldInfo */
class AttrFieldInfo : public ObjectRef {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AttrFieldInfo, ObjectRef, AttrFieldInfoNode);
};

/*!
 * \brief Base class of all attribute class
 * \note Do not subclass AttrBaseNode directly,
 *       subclass AttrsNode instead.
 * \sa AttrsNode
 */
class BaseAttrsNode : public Object {
 public:
  /*! \brief virtual destructor */
  virtual ~BaseAttrsNode() {
  }
  // visit function
  virtual void VisitAttrs(AttrVisitor* v) {
  }
  /*!
   * \brief Print readible docstring to ostream, add newline.
   * \param os the stream to print the docstring to.
   */
  inline void PrintDocString(std::ostream& os) const;  // NOLINT(*)
  /*!
   * \brief Visit attributes that do not equal the default value.
   *
   * \note This is useful to extract fields for concise printing.
   * \param v The visitor
   */
  MATX_DLL virtual void VisitNonDefaultAttrs(AttrVisitor* v) = 0;
  /*!
   * \brief Get the field information
   * \return The fields in the Attrs.
   */
  MATX_DLL virtual runtime::Array<AttrFieldInfo> ListFieldInfo() const = 0;

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "Attrs";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(BaseAttrsNode, Object);
};

/*!
 * \brief Managed reference to BaseAttrsNode.
 * \sa AttrsNode, BaseAttrsNode
 */
class Attrs : public ObjectRef {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Attrs, ObjectRef, BaseAttrsNode);
};

/*!
 * \brief Specialized attribute type that is backed by a map.
 *  The DictAttrsNode implements the Attrs behavior,
 *  its fields are directly accessible via object.field_name
 *  like other normal nodes.
 */
class DictAttrsNode : public BaseAttrsNode {
 public:
  /*! \brief internal attrs map */
  runtime::Map<StringRef, ObjectRef> dict;

  bool SEqualReduce(const DictAttrsNode* other, SEqualReducer equal) const {
    return equal(dict, other->dict);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dict);
  }

  // implementations
  void VisitAttrs(AttrVisitor* v) final;
  void VisitNonDefaultAttrs(AttrVisitor* v) final;
  runtime::Array<AttrFieldInfo> ListFieldInfo() const final;
  // type info
  static constexpr const char* _type_key = "DictAttrs";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DictAttrsNode, BaseAttrsNode);
};

/*!
 * \brief Managed reference to DictAttrsNode
 * \sa DictAttrsNode.
 */
class DictAttrs : public Attrs {
 public:
  /*!
   * \brief Consruct a Attrs backed by DictAttrsNode.
   * \param dict The attributes.
   * \return The dict attributes.
   */
  MATX_DLL explicit DictAttrs(runtime::Map<StringRef, ObjectRef> dict);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(DictAttrs, Attrs, DictAttrsNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(DictAttrsNode);
};

// Namespace containing detail implementations
namespace detail {

// helper entry that does nothing in set_default/bound/describe calls.
struct AttrNopEntry {
  using TSelf = AttrNopEntry;

  TSelf& describe(MATXSCRIPT_ATTRIBUTE_UNUSED const char* str) {
    return *this;
  }
  template <typename T>
  TSelf& set_default(MATXSCRIPT_ATTRIBUTE_UNUSED const T& value) {
    return *this;
  }
  template <typename T>
  TSelf& set_lower_bound(MATXSCRIPT_ATTRIBUTE_UNUSED const T& begin) {
    return *this;
  }
  template <typename T>
  TSelf& set_upper_bound(MATXSCRIPT_ATTRIBUTE_UNUSED const T& end) {
    return *this;
  }
};

// Wrapper for normal visitor.
class AttrNormalVisitor {
 public:
  explicit AttrNormalVisitor(AttrVisitor* visitor) : visitor_(visitor) {
  }
  template <typename T>
  AttrNopEntry operator()(const char* key, T* value) {
    visitor_->Visit(key, value);
    return AttrNopEntry();
  }

 private:
  AttrVisitor* visitor_;
};

class AttrsSEqualVisitor {
 public:
  bool result_{true};
  // constructor
  AttrsSEqualVisitor(const Object* lhs, const Object* rhs, const SEqualReducer& equal)
      : lhs_(lhs), rhs_(rhs), equal_(equal) {
  }
  template <typename T>
  AttrNopEntry operator()(const char* key, T* lhs_value) {
    if (!result_)
      return AttrNopEntry();
    const T* rhs_value = reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(rhs_) +
        (reinterpret_cast<const char*>(lhs_value) - reinterpret_cast<const char*>(lhs_)));
    if (!equal_(*lhs_value, *rhs_value)) {
      result_ = false;
    }
    return AttrNopEntry();
  }

 private:
  const Object* lhs_;
  const Object* rhs_;
  const SEqualReducer& equal_;
};

class AttrsSHashVisitor {
 public:
  explicit AttrsSHashVisitor(const SHashReducer& hash_reducer) : hash_reducer_(hash_reducer) {
  }

  template <typename T>
  AttrNopEntry operator()(const char* key, T* value) {
    hash_reducer_(*value);
    return AttrNopEntry();
  }

 private:
  const SHashReducer& hash_reducer_;
};

/*!
 * \brief Helper struct to get the type name known to matxscript.
 * \tparam T the type we are interested in.
 */
template <typename T>
struct TypeName {
  static constexpr const char* value = T::ContainerType::_type_key;
};

template <>
struct TypeName<int> {
  static constexpr const char* value = "int";
};

template <>
struct TypeName<int64_t> {
  static constexpr const char* value = "int64";
};

template <>
struct TypeName<uint64_t> {
  static constexpr const char* value = "uint64_t";
};

template <>
struct TypeName<runtime::DataType> {
  static constexpr const char* value = "DataType";
};

template <>
struct TypeName<std::string> {
  static constexpr const char* value = "str";
};

template <>
struct TypeName<bool> {
  static constexpr const char* value = "bool";
};

template <>
struct TypeName<void*> {
  static constexpr const char* value = "handle";
};

template <>
struct TypeName<double> {
  static constexpr const char* value = "double";
};

class AttrDocEntry {
 public:
  using TSelf = AttrDocEntry;

  explicit AttrDocEntry(ObjectPtr<AttrFieldInfoNode> info) : info_(info) {
  }
  TSelf& describe(const char* str) {
    info_->description = str;
    return *this;
  }
  template <typename T>
  TSelf& set_default(const T& value) {
    std::ostringstream os;
    os << info_->type_info << ", default=" << value;
    info_->type_info = os.str();
    return *this;
  }
  template <typename T>
  TSelf& set_lower_bound(MATXSCRIPT_ATTRIBUTE_UNUSED T begin) {
    return *this;
  }
  template <typename T>
  TSelf& set_upper_bound(MATXSCRIPT_ATTRIBUTE_UNUSED T end) {
    return *this;
  }

 private:
  ObjectPtr<AttrFieldInfoNode> info_;
};

class AttrDocVisitor {
 public:
  template <typename T>
  AttrDocEntry operator()(const char* key, T* v) {
    ObjectPtr<AttrFieldInfoNode> info = runtime::make_object<AttrFieldInfoNode>();
    info->name = key;
    info->type_info = TypeName<T>::value;
    fields_.push_back(AttrFieldInfo(info));
    return AttrDocEntry(info);
  }

  runtime::Array<AttrFieldInfo> fields_;
};

class AttrExistVisitor {
 public:
  std::string key_;
  bool exist_{false};

  template <typename T>
  AttrNopEntry operator()(const char* key, T* v) {
    if (exist_)
      return AttrNopEntry();
    if (key == key_)
      exist_ = true;
    return AttrNopEntry();
  }
};

template <typename T>
struct AttrTriggerNonDefaultEntry {
  using TSelf = AttrTriggerNonDefaultEntry<T>;
  // constructor
  AttrTriggerNonDefaultEntry(AttrVisitor* visitor, const char* key, T* data)
      : visitor_(visitor), key_(key), data_(data) {
  }

  ~AttrTriggerNonDefaultEntry() MATXSCRIPT_THROW_EXCEPTION {
    if (trigger_) {
      visitor_->Visit(key_, data_);
    }
  }
  TSelf& describe(MATXSCRIPT_ATTRIBUTE_UNUSED const char* str) {
    return *this;
  }
  TSelf& set_default(const T& value) {
    if (runtime::StructuralEqual()(value, *data_)) {
      trigger_ = false;
    }
    return *this;
  }
  TSelf& set_lower_bound(MATXSCRIPT_ATTRIBUTE_UNUSED const T& begin) {
    return *this;
  }
  TSelf& set_upper_bound(MATXSCRIPT_ATTRIBUTE_UNUSED const T& end) {
    return *this;
  }

 private:
  AttrVisitor* visitor_;
  const char* key_;
  T* data_;
  bool trigger_{true};
};

class AttrNonDefaultVisitor {
 public:
  explicit AttrNonDefaultVisitor(AttrVisitor* visitor) : visitor_(visitor) {
  }
  template <typename T>
  AttrTriggerNonDefaultEntry<T> operator()(const char* key, T* value) {
    return AttrTriggerNonDefaultEntry<T>(visitor_, key, value);
  }

 private:
  AttrVisitor* visitor_;
};
}  // namespace detail

/*!
 * \brief The base class of the all the
 *  Use "curiously recurring template pattern".
 *
 * \tparam DerivedType The final attribute type.
 */
template <typename DerivedType>
class AttrsNode : public BaseAttrsNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    ::matxscript::ir::detail::AttrNormalVisitor vis(v);
    self()->__VisitAttrs__(vis);
  }

  void VisitNonDefaultAttrs(AttrVisitor* v) {
    ::matxscript::ir::detail::AttrNonDefaultVisitor vis(v);
    self()->__VisitAttrs__(vis);
  }

  bool SEqualReduce(const DerivedType* other, SEqualReducer equal) const {
    DerivedType* pself = self();
    ::matxscript::ir::detail::AttrsSEqualVisitor visitor(pself, other, equal);
    self()->__VisitAttrs__(visitor);
    return visitor.result_;
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    ::matxscript::ir::detail::AttrsSHashVisitor visitor(hash_reducer);
    self()->__VisitAttrs__(visitor);
  }

  runtime::Array<AttrFieldInfo> ListFieldInfo() const final {
    ::matxscript::ir::detail::AttrDocVisitor visitor;
    self()->__VisitAttrs__(visitor);
    return visitor.fields_;
  }

 private:
  DerivedType* self() const {
    return const_cast<DerivedType*>(static_cast<const DerivedType*>(this));
  }
};

inline void BaseAttrsNode::PrintDocString(std::ostream& os) const {  // NOLINT(*)
  runtime::Array<AttrFieldInfo> entry = this->ListFieldInfo();
  for (AttrFieldInfo info : entry) {
    os << info->name << " : " << info->type_info << '\n';
    if (info->description.length() != 0) {
      os << "    " << info->description << '\n';
    }
  }
}

}  // namespace ir
}  // namespace matxscript
