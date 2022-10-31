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
#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/runtime_port.h>

/*!
 * \brief Whether or not use atomic reference counter.
 *  If the reference counter is not atomic,
 *  an object cannot be owned by multiple threads.
 *  We can, however, move an object across threads
 */
#ifndef MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER
#define MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER 1
#endif

#if MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER
#include <atomic>
#endif  // MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER

namespace matxscript {
namespace runtime {

/*!
 * \brief base class of all object containers.
 *
 * Sub-class of objects should declare the following static constexpr fields:
 *
 * - _type_index:
 *      Static type index of the object, if assigned to TypeIndex::kDynamic
 *      the type index will be assigned during runtime.
 *      Runtime type index can be accessed by ObjectType::TypeIndex();
 * - _type_key:
 *       The unique string identifier of the type.
 * - _type_final:
 *       Whether the type is terminal type(there is no subclass of the type in the object system).
 *       This field is automatically set by macro MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO
 *       It is still OK to sub-class a terminal object type T and construct it using make_object.
 *       But IsInstance check will only show that the object type is T(instead of the sub-class).
 *
 * The following two fields are necessary for base classes that can be sub-classed.
 *
 * - _type_child_slots:
 *       Number of reserved type index slots for child classes.
 *       Used for runtime optimization for type checking in IsInstance.
 *       If an object's type_index is within range of [type_index, type_index + _type_child_slots]
 *       Then the object can be quickly decided as sub-class of the current object class.
 *       If not, a fallback mechanism is used to check the global type table.
 *       Recommendation: set to estimate number of children needed.
 * - _type_child_slots_can_overflow:
 *       Whether we can add additional child classes even if the number of child classes
 *       exceeds the _type_child_slots. A fallback mechanism to check global type table will be
 * used. Recommendation: set to false for optimal runtime speed if we know exact number of children.
 *
 * Two macros are used to declare helper functions in the object:
 * - Use MATXSCRIPT_DECLARE_BASE_OBJECT_INFO for object classes that can be sub-classed.
 * - Use MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO for object classes that cannot be sub-classed.
 *
 * New objects can be created using make_object function.
 * Which will automatically populate the type_index and deleter of the object.
 *
 * \sa make_object
 * \sa ObjectPtr
 * \sa ObjectRef
 *
 * \code
 *
 *  // Create a base object
 *  class BaseObj : public Object {
 *   public:
 *    // object fields
 *    int field0;
 *
 *    // object properties
 *    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
 *    static constexpr const char* _type_key = "test.BaseObj";
 *    MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(BaseObj, Object);
 *  };
 *
 *  class ObjLeaf : public ObjBase {
 *   public:
 *    // fields
 *    int child_field0;
 *    // object properties
 *    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
 *    static constexpr const char* _type_key = "test.LeafObj";
 *    MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(LeafObj, Object);
 *  };
 *
 *  // The following code should be put into a cc file.
 *  MATXSCRIPT_REGISTER_OBJECT_TYPE(ObjBase);
 *  MATXSCRIPT_REGISTER_OBJECT_TYPE(ObjLeaf);
 *
 *  // Usage example.
 *  void TestObjects() {
 *    // create an object
 *    ObjectRef leaf_ref(make_object<LeafObj>());
 *    // cast to a specific instance
 *    const LeafObj* leaf_ptr = leaf_ref.as<LeafObj>();
 *    CHECK(leaf_ptr != nullptr);
 *    // can also cast to the base class.
 *    CHECK(leaf_ref.as<BaseObj>() != nullptr);
 *  }
 *
 * \endcode
 */
class Object {
 public:
  /*!
   * \brief Object deleter
   * \param self pointer to the Object.
   */
  typedef void (*FDeleter)(Object* self);
  /*! \return The internal runtime type index of the object. */
  uint32_t type_index() const noexcept {
    return type_index_;
  }
  /*!
   * \return the type key of the object.
   * \note this operation is expensive, can be used for error reporting.
   */
  string_view GetTypeKey() const {
    return TypeIndex2Key(type_index_);
  }
  /*!
   * \return A hash value of the return of GetTypeKey.
   */
  size_t GetTypeKeyHash() const {
    return TypeIndex2KeyHash(type_index_);
  }
  /*!
   * Check if the object is an instance of TargetType.
   * \tparam TargetType The target type to be checked.
   * \return Whether the target type is true.
   */
  template <typename TargetType>
  inline bool IsInstance() const;

  /*!
   * \brief Get the type key of the corresponding index from runtime.
   * \param tindex The type index.
   * \return the result.
   */
  MATX_DLL static string_view TypeIndex2Key(uint32_t tindex);
  MATX_DLL static bool TryTypeIndex2Key(uint32_t tindex, string_view* tkey);
  /*!
   * \brief Get the type key hash of the corresponding index from runtime.
   * \param tindex The type index.
   * \return the related key-hash.
   */
  MATX_DLL static size_t TypeIndex2KeyHash(uint32_t tindex);
  /*!
   * \brief Get the type index of the corresponding key from runtime.
   * \param key The type key.
   * \return the result.
   */
  MATX_DLL static uint32_t TypeKey2Index(const string_view& key);

#if MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER
  using RefCounterType = std::atomic<int32_t>;
#else
  using RefCounterType = int32_t;
#endif

  static constexpr const char* _type_key = "runtime.Object";

  static uint32_t _GetOrAllocRuntimeTypeIndex() {
    return TypeIndex::kRoot;
  }
  static uint32_t RuntimeTypeIndex() {
    return TypeIndex::kRoot;
  }

  // Default object type properties for sub-classes
  static constexpr bool _type_final = false;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_child_slots_can_overflow = true;
  // member information
  static constexpr bool _type_has_method_visit_attrs = true;
  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  // NOTE: the following field is not type index of Object
  // but was intended to be used by sub-classes as default value.
  // The type index of Object is TypeIndex::kRoot
  static constexpr uint32_t _type_index = TypeIndex::kDynamic;

  // Default constructor and copy constructor
  Object() noexcept {
  }
  // Override the copy and assign constructors to do nothing.
  // This is to make sure only contents, but not deleter and ref_counter
  // are copied when a child class copies itself.
  // This will enable us to use make_object<ObjectClass>(*obj_ptr)
  // to copy an existing object.
  Object(const Object& other) noexcept {
  }
  Object(Object&& other) noexcept {
  }
  Object& operator=(const Object& other) noexcept {
    return *this;
  }
  Object& operator=(Object&& other) noexcept {
    return *this;
  }

 protected:
  // The fields of the base object cell.
  /*! \brief Type index(tag) that indicates the type of the object. */
  uint32_t type_index_{0};
  /*! \brief The internal reference counter */
  RefCounterType ref_counter_{1};
  /*!
   * \brief deleter of this object to enable customized allocation.
   * If the deleter is nullptr, no deletion will be performed.
   * The creator of the object must always set the deleter field properly.
   */
  FDeleter deleter_ = nullptr;
  // Invariant checks.
  static_assert(sizeof(int32_t) == sizeof(RefCounterType) &&
                    alignof(int32_t) == sizeof(RefCounterType),
                "RefCounter ABI check.");

  /*!
   * \brief Get the type index using type key.
   *
   *  When the function is first time called for a type,
   *  it will register the type to the type table in the runtime.
   *  If the static_tindex is TypeIndex::kDynamic, the function will
   *  allocate a runtime type index.
   *  Otherwise, we will populate the type table and return the static index.
   *
   * \param key the type key.
   * \param static_tindex The current _type_index field.
   *                      can be TypeIndex::kDynamic.
   * \param parent_tindex The index of the parent.
   * \param type_child_slots Number of slots reserved for its children.
   * \param type_child_slots_can_overflow Whether to allow child to overflow the slots.
   * \return The allocated type index.
   */
  MATX_DLL static uint32_t GetOrAllocRuntimeTypeIndex(const string_view& key,
                                                      uint32_t static_tindex,
                                                      uint32_t parent_tindex,
                                                      uint32_t type_child_slots,
                                                      bool type_child_slots_can_overflow);

  // reference counter related operations
  /*! \brief developer function, increases reference counter. */
  inline void IncRef() noexcept;
  /*!
   * \brief developer function, decrease reference counter.
   * \note The deleter will be called when ref_counter_ becomes zero.
   */
  inline void DecRef() noexcept;

 private:
  /*!
   * \return The usage count of the cell.
   * \note We use stl style naming to be consistent with known API in shared_ptr.
   */
  inline int use_count() const noexcept;
  /*!
   * \brief Check of this object is derived from the parent.
   * \param parent_tindex The parent type index.
   * \return The derivation results.
   */
  MATX_DLL bool DerivedFrom(uint32_t parent_tindex) const;
  // friend classes
  template <typename>
  friend class ObjAllocatorBase;
  template <typename>
  friend class ObjectPtr;
  friend class ObjectInternal;
  friend class ObjectHash;
  friend class ObjectEqual;
  friend class RTValue;
  friend class RTView;
  template <typename>
  friend struct ObjectView;
};

/*!
 * \brief Get a reference type from a raw object ptr type
 *
 *  It is always important to get a reference type
 *  if we want to return a value as reference or keep
 *  the object alive beyond the scope of the function.
 *
 * \param ptr The object pointer
 * \tparam RefType The reference type
 * \tparam ObjectType The object type
 * \return The corresponding RefType
 */
template <typename ObjectRefType, typename ObjectType>
inline ObjectRefType GetRef(const ObjectType* ptr);

/*!
 * \brief Downcast a base reference type to a more specific type.
 *
 * \param ref The inptut reference
 * \return The corresponding SubRef.
 * \tparam SubRef The target specific reference type.
 * \tparam BaseRef the current reference type.
 */
template <typename SubRef, typename BaseRef>
inline SubRef Downcast(BaseRef ref);

template <typename>
struct ObjectView;

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class ObjectPtr {
 public:
  /*! \brief default constructor */
  ObjectPtr() noexcept {
  }
  /*! \brief default constructor */
  ObjectPtr(std::nullptr_t) noexcept {
  }
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  ObjectPtr(const ObjectPtr<T>& other) noexcept : ObjectPtr(other.data_) {
  }
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template <typename U>
  ObjectPtr(const ObjectPtr<U>& other) noexcept : ObjectPtr(other.data_) {
    static_assert(std::is_base_of<T, U>::value,
                  "can only assign of child class ObjectPtr to parent");
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  ObjectPtr(ObjectPtr<T>&& other) noexcept : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  template <typename Y>
  ObjectPtr(ObjectPtr<Y>&& other) noexcept : data_(other.data_) {
    static_assert(std::is_base_of<T, Y>::value,
                  "can only assign of child class ObjectPtr to parent");
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~ObjectPtr() {
    this->reset();
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(ObjectPtr<T>& other) noexcept {
    std::swap(data_, other.data_);
  }
  /*!
   * \return Get the content of the pointer
   */
  T* get() const noexcept {
    return static_cast<T*>(data_);
  }
  /*!
   * \return The pointer
   */
  T* operator->() const noexcept {
    return get();
  }
  /*!
   * \return The reference
   */
  T& operator*() const noexcept {
    return *get();
  }
  /*!
   * \brief copy assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(const ObjectPtr<T>& other) noexcept {
    // takes in plane operator to enable copy elison.
    // copy-and-swap idiom
    ObjectPtr(other).swap(*this);
    return *this;
  }
  /*!
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(ObjectPtr<T>&& other) noexcept {
    // copy-and-swap idiom
    ObjectPtr(std::move(other)).swap(*this);
    return *this;
  }
  /*! \brief reset the content of ptr to be nullptr */
  void reset() noexcept {
    if (data_ != nullptr) {
      data_->DecRef();
      data_ = nullptr;
    }
  }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const noexcept {
    return data_ != nullptr ? data_->use_count() : 0;
  }
  /*! \return whether the reference is unique */
  bool unique() const noexcept {
    return data_ != nullptr && data_->use_count() == 1;
  }
  /*! \return Whether two ObjectPtr do not equal each other */
  bool operator==(const ObjectPtr<T>& other) const noexcept {
    return data_ == other.data_;
  }
  /*! \return Whether two ObjectPtr equals each other */
  bool operator!=(const ObjectPtr<T>& other) const noexcept {
    return data_ != other.data_;
  }
  /*! \return Whether the pointer is nullptr */
  bool operator==(std::nullptr_t null) const noexcept {
    return data_ == nullptr;
  }
  /*! \return Whether the pointer is not nullptr */
  bool operator!=(std::nullptr_t null) const noexcept {
    return data_ != nullptr;
  }

 private:
  /*! \brief internal pointer field */
  Object* data_{nullptr};
  /*!
   * \brief constructor from Object
   * \param data The data pointer
   */
  explicit ObjectPtr(Object* data) noexcept : data_(data) {
    if (data != nullptr) {
      data_->IncRef();
    }
  }
  /*!
   * \brief Move an ObjectPtr from an RValueRef argument.
   * \param ref The rvalue reference.
   * \return the moved result.
   */
  static ObjectPtr<T> MoveFromRValueRefArg(Object** ref) noexcept {
    ObjectPtr<T> ptr;
    ptr.data_ = *ref;
    *ref = nullptr;
    return ptr;
  }
  // friend classes
  friend class Object;
  friend class ObjectRef;
  friend struct ObjectPtrHash;
  template <typename>
  friend class ObjectPtr;
  template <typename>
  friend class ObjAllocatorBase;
  friend class RTValue;
  friend class RTView;
  template <typename ObjectRefType, typename ObjType>
  friend ObjectRefType GetRef(const ObjType* ptr);
  template <typename BaseType, typename ObjType>
  friend ObjectPtr<BaseType> GetObjectPtr(ObjType* ptr) noexcept;
  template <typename>
  friend struct ObjectView;
};

/*! \brief Base class of all object reference */
class ObjectRef {
 public:
  /*! \brief default constructor */
  ObjectRef() noexcept = default;
  /*! \brief Constructor from existing object ptr */
  explicit ObjectRef(ObjectPtr<Object> data) noexcept : data_(std::move(data)) {
  }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool same_as(const ObjectRef& other) const noexcept {
    return data_ == other.data_;
  }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator==(const ObjectRef& other) const noexcept {
    return data_ == other.data_;
  }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator!=(const ObjectRef& other) const noexcept {
    return data_ != other.data_;
  }
  /*!
   * \brief Comparator
   * \param other Another object ref by address.
   * \return the compare result.
   */
  bool operator<(const ObjectRef& other) const noexcept {
    return data_.get() < other.data_.get();
  }
  /*!
   * \return whether the object is defined(not null).
   */
  bool defined() const noexcept {
    return data_ != nullptr;
  }
  /*! \return the internal object pointer */
  const Object* get() const noexcept {
    return data_.get();
  }
  /*! \return the internal object pointer */
  const Object* operator->() const noexcept {
    return get();
  }
  /*! \return whether the reference is unique */
  bool unique() const noexcept {
    return data_.unique();
  }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const noexcept {
    return data_.use_count();
  }
  /*!
   * \brief Try to downcast the internal Object to a
   *  raw pointer of a corresponding type.
   *
   *  The function will return a nullptr if the cast failed.
   *
   * if (const Add *add = node_ref.As<Add>()) {
   *   // This is an add node
   * }
   * \tparam ObjectType the target type, must be a subtype of Object/
   */
  template <typename ObjectType>
  inline const ObjectType* as() const noexcept;

  /*! \brief type indicate the container type. */
  using ContainerType = Object;
  // Default type properties for the reference class.
  static constexpr bool _type_is_nullable = true;

 protected:
  /*! \brief Internal pointer that backs the reference. */
  ObjectPtr<Object> data_;
  /*! \return return a mutable internal ptr, can be used by sub-classes. */
  Object* get_mutable() const noexcept {
    return data_.get();
  }
  /*!
   * \brief Internal helper function downcast a ref without check.
   * \note Only used for internal dev purposes.
   * \tparam T The target reference type.
   * \return The casted result.
   */
  template <typename T>
  static T DowncastNoCheck(ObjectRef ref) noexcept {
    return T(std::move(ref.data_));
  }
  /*!
   * \brief Clear the object ref data field without DecRef
   *        after we successfully moved the field.
   * \param ref The reference data.
   */
  static void FFIClearAfterMove(ObjectRef* ref) noexcept {
    ref->data_.data_ = nullptr;
  }
  /*!
   * \brief Internal helper function get data_ as ObjectPtr of ObjectType.
   * \note only used for internal dev purpose.
   * \tparam ObjectType The corresponding object type.
   * \return the corresponding type.
   */
  static ObjectPtr<Object> GetDataPtr(const ObjectRef& ref) noexcept {
    return ref.data_;
  }
  // friend classes.
  friend struct ObjectPtrHash;
  friend class TRetValue;
  friend class TArgsSetter;
  friend class ObjectInternal;
  friend class RTValue;
  friend class RTView;
  template <typename SubRef, typename BaseRef>
  friend SubRef Downcast(BaseRef ref);
  template <typename SubRef, typename BaseRef>
  friend SubRef DowncastNoCheck(BaseRef ref) noexcept;
  template <typename>
  friend struct ObjectView;
};

#define MX_DPTR(Class) Class##Node* d = static_cast<Class##Node*>(data_.get())
#define MX_CHECK_DPTR(ClassName) \
  MX_DPTR(ClassName);            \
  MXCHECK(d != nullptr) << "[" << #ClassName << "] object is None"
#define MX_QPTR(Class) Class* q = static_cast<Class*>(q_ptr)

/*!
 * \brief Get an object ptr type from a raw object ptr.
 *
 * \param ptr The object pointer
 * \tparam BaseType The reference type
 * \tparam ObjectType The object type
 * \return The corresponding RefType
 */
template <typename BaseType, typename ObjectType>
inline ObjectPtr<BaseType> GetObjectPtr(ObjectType* ptr) noexcept;

/*! \brief ObjectRef hash functor */
struct ObjectPtrHash {
  size_t operator()(const ObjectRef& a) const noexcept {
    return operator()(a.data_);
  }

  template <typename T>
  size_t operator()(const ObjectPtr<T>& a) const noexcept {
    return std::hash<Object*>()(a.get());
  }
};

/*! \brief ObjectRef equal functor */
struct ObjectPtrEqual {
  bool operator()(const ObjectRef& a, const ObjectRef& b) const noexcept {
    return a.same_as(b);
  }

  template <typename T>
  size_t operator()(const ObjectPtr<T>& a, const ObjectPtr<T>& b) const noexcept {
    return a == b;
  }
};

/*!
 * \brief helper macro to declare a base object type that can be inheritated.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)                           \
  static_assert(!ParentType::_type_final, "ParentObj maked as final");                      \
  static uint32_t RuntimeTypeIndex() {                                                      \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 || \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,          \
                  "Need to set _type_child_slots when parent specifies it.");               \
    if (TypeName::_type_index != ::matxscript::runtime::TypeIndex::kDynamic) {              \
      return TypeName::_type_index;                                                         \
    }                                                                                       \
    return _GetOrAllocRuntimeTypeIndex();                                                   \
  }                                                                                         \
  static uint32_t _GetOrAllocRuntimeTypeIndex() {                                           \
    static uint32_t tidx =                                                                  \
        Object::GetOrAllocRuntimeTypeIndex(TypeName::_type_key,                             \
                                           TypeName::_type_index,                           \
                                           ParentType::_GetOrAllocRuntimeTypeIndex(),       \
                                           TypeName::_type_child_slots,                     \
                                           TypeName::_type_child_slots_can_overflow);       \
    return tidx;                                                                            \
  }

/*!
 * \brief helper macro to declare type information in a final class.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TypeName, ParentType) \
  static const constexpr bool _type_final = true;                  \
  static const constexpr int _type_child_slots = 0;                \
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)

#define MATXSCRIPT_OBJECT_REG_VAR_DEF static MATXSCRIPT_ATTRIBUTE_UNUSED uint32_t __make_Object_tid

/*!
 * \brief Helper macro to register the object type to runtime.
 *  Makes sure that the runtime type table is correctly populated.
 *
 *  Use this macro in the cc file for each terminal class.
 */
#define MATXSCRIPT_REGISTER_OBJECT_TYPE(TypeName)                     \
  MATXSCRIPT_STR_CONCAT(MATXSCRIPT_OBJECT_REG_VAR_DEF, __COUNTER__) = \
      TypeName::_GetOrAllocRuntimeTypeIndex()

/*
 * \brief Define object reference methods.
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 */
#define MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)                  \
  TypeName() noexcept = default;                                                                \
  explicit TypeName(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n) noexcept \
      : ParentType(std::move(n)) {                                                              \
  }                                                                                             \
  TypeName(const TypeName& other) noexcept = default;                                           \
  TypeName(TypeName&& other) noexcept = default;                                                \
  TypeName& operator=(const TypeName& other) noexcept = default;                                \
  TypeName& operator=(TypeName&& other) noexcept = default;                                     \
  const ObjectName* operator->() const noexcept {                                               \
    return static_cast<const ObjectName*>(data_.get());                                         \
  }                                                                                             \
  const ObjectName* get() const noexcept {                                                      \
    return operator->();                                                                        \
  }                                                                                             \
  using ContainerType = ObjectName;

/*
 * \brief Define object reference methods that is not nullable.
 *
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 */
#define MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)      \
  explicit TypeName(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n) noexcept \
      : ParentType(std::move(n)) {                                                              \
  }                                                                                             \
  TypeName(const TypeName& other) noexcept = default;                                           \
  TypeName(TypeName&& other) noexcept = default;                                                \
  TypeName& operator=(const TypeName& other) noexcept = default;                                \
  TypeName& operator=(TypeName&& other) noexcept = default;                                     \
  const ObjectName* operator->() const noexcept {                                               \
    return static_cast<const ObjectName*>(data_.get());                                         \
  }                                                                                             \
  const ObjectName* get() const noexcept {                                                      \
    return operator->();                                                                        \
  }                                                                                             \
  static constexpr bool _type_is_nullable = false;                                              \
  using ContainerType = ObjectName;

/*
 * \brief Define object reference methods of whose content is mutable.
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 * \note We recommend making objects immutable when possible.
 *       This macro is only reserved for objects that stores runtime states.
 */
#define MATXSCRIPT_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)          \
  TypeName() noexcept = default;                                                                \
  TypeName(const TypeName& other) noexcept = default;                                           \
  TypeName(TypeName&& other) noexcept = default;                                                \
  TypeName& operator=(const TypeName& other) noexcept = default;                                \
  TypeName& operator=(TypeName&& other) noexcept = default;                                     \
  explicit TypeName(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n) noexcept \
      : ParentType(std::move(n)) {                                                              \
  }                                                                                             \
  ObjectName* operator->() const noexcept {                                                     \
    return static_cast<ObjectName*>(data_.get());                                               \
  }                                                                                             \
  using ContainerType = ObjectName;

/*!
 * \brief Define CopyOnWrite function in an ObjectRef.
 * \param ObjectName The Type of the Node.
 *
 *  CopyOnWrite will generate a unique copy of the internal node.
 *  The node will be copied if it is referenced by multiple places.
 *  The function returns the raw pointer to the node to allow modification
 *  of the content.
 *
 * \code
 *
 *  MyCOWObjectRef ref, ref2;
 *  ref2 = ref;
 *  ref.CopyOnWrite()->value = new_value;
 *  assert(ref2->value == old_value);
 *  assert(ref->value == new_value);
 *
 * \endcode
 */
#define MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ObjectName)                                      \
  ObjectName* CopyOnWrite() {                                                                    \
    MXCHECK(data_ != nullptr);                                                                   \
    if (!data_.unique()) {                                                                       \
      auto n = ::matxscript::runtime::make_object<ObjectName>(*(operator->()));                  \
      ::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object>(std::move(n)).swap(data_); \
    }                                                                                            \
    return static_cast<ObjectName*>(data_.get());                                                \
  }

// Implementations details below
// Object reference counting.
#if MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER

inline void Object::IncRef() noexcept {
  ref_counter_.fetch_add(1, std::memory_order_relaxed);
}

inline void Object::DecRef() noexcept {
  if (use_count() == 1) {
    if (this->deleter_ != nullptr) {
      (*this->deleter_)(this);
    }
  } else if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    if (this->deleter_ != nullptr) {
      (*this->deleter_)(this);
    }
  }
}

inline int Object::use_count() const noexcept {
  return ref_counter_.load(std::memory_order_relaxed);
}

#else

inline void Object::IncRef() noexcept {
  ++ref_counter_;
}

inline void Object::DecRef() noexcept {
  if (--ref_counter_ == 0) {
    if (this->deleter_ != nullptr) {
      (*this->deleter_)(this);
    }
  }
}

inline int Object::use_count() const noexcept {
  return ref_counter_;
}

#endif  // MATXSCRIPT_OBJECT_ATOMIC_REF_COUNTER

template <typename TargetType>
inline bool Object::IsInstance() const {
  const Object* self = this;
  // NOTE: the following code can be optimized by
  // compiler dead-code elimination for already known constants.
  if (self != nullptr) {
    // Everything is a subclass of object.
    if (std::is_same<TargetType, Object>::value)
      return true;
    if (TargetType::_type_final) {
      // if the target type is a final type
      // then we only need to check the equivalence.
      return self->type_index_ == TargetType::RuntimeTypeIndex();
    } else {
      // if target type is a non-leaf type
      // Check if type index falls into the range of reserved slots.
      uint32_t begin = TargetType::RuntimeTypeIndex();
      // The condition will be optimized by constant-folding.
      if (TargetType::_type_child_slots != 0) {
        uint32_t end = begin + TargetType::_type_child_slots;
        if (self->type_index_ >= begin && self->type_index_ < end)
          return true;
      } else {
        if (self->type_index_ == begin)
          return true;
      }
      if (!TargetType::_type_child_slots_can_overflow)
        return false;
      // Invariance: parent index is always smaller than the child.
      if (self->type_index_ < TargetType::RuntimeTypeIndex())
        return false;
      // The rare slower-path, check type hierachy.
      return self->DerivedFrom(TargetType::RuntimeTypeIndex());
    }
  } else {
    return false;
  }
}

template <typename ObjectType>
inline const ObjectType* ObjectRef::as() const noexcept {
  if (data_ != nullptr && data_->IsInstance<ObjectType>()) {
    return static_cast<ObjectType*>(data_.get());
  } else {
    return nullptr;
  }
}

template <typename RefType, typename ObjType>
inline RefType GetRef(const ObjType* ptr) {
  static_assert(std::is_base_of<typename RefType::ContainerType, ObjType>::value,
                "Can only cast to the ref of same container type");
  if (!RefType::_type_is_nullable) {
    MXCHECK(ptr != nullptr);
  }
  return RefType(ObjectPtr<Object>(const_cast<Object*>(static_cast<const Object*>(ptr))));
}

template <typename BaseType, typename ObjType>
inline ObjectPtr<BaseType> GetObjectPtr(ObjType* ptr) noexcept {
  static_assert(std::is_base_of<BaseType, ObjType>::value,
                "Can only cast to the ref of same container type");
  return ObjectPtr<BaseType>(static_cast<Object*>(ptr));
}

template <typename TObjectRef>
inline bool IsConvertible(const Object* node) {
  if (std::is_base_of<ObjectRef, TObjectRef>::value) {
    return node ? node->IsInstance<typename TObjectRef::ContainerType>()
                : TObjectRef::_type_is_nullable;
  }
  return false;
}

template <typename SubRef, typename BaseRef>
inline SubRef Downcast(BaseRef ref) {
  if (ref.defined()) {
    MXCHECK(ref->template IsInstance<typename SubRef::ContainerType>())
        << "Downcast from " << ref->GetTypeKey() << " to " << SubRef::ContainerType::_type_key
        << " failed.";
  } else {
    MXCHECK(SubRef::_type_is_nullable) << "Downcast from nullptr to not nullable reference of "
                                       << SubRef::ContainerType::_type_key;
  }
  return SubRef(std::move(ref.data_));
}

template <typename SubRef, typename BaseRef>
inline SubRef DowncastNoCheck(BaseRef ref) noexcept {
  return SubRef(std::move(ref.data_));
}

template <typename SubRef, typename>
inline SubRef DowncastNoCheck(ObjectRef ref) noexcept {
  return SubRef(std::move(ref.data_));
}

namespace TypeIndex {

template <typename T>
struct object_has_container_type {
  template <typename other>
  static char judge(typename other::ContainerType* x) {
    return 0;
  };
  template <typename other>
  static int judge(...) {
    return 1;
  };
  constexpr static bool value = sizeof(judge<T>(0)) == sizeof(char);
};

template <bool, typename TYPE>
struct traits_helper_step2;

template <typename TYPE>
struct traits_helper_step2<true, TYPE> {
  static inline int32_t GetRuntimeIndex() {
    return TYPE::ContainerType::RuntimeTypeIndex();
  }
};

template <typename TYPE>
struct traits_helper_step2<false, TYPE> {
  static inline int32_t GetRuntimeIndex() {
    MXTHROW << "unknown type index: " << typeid(TYPE).name();
    return kRuntimeUnknown;
  }
};

template <bool, typename TYPE>
struct traits_helper_step1;

template <typename TYPE>
struct traits_helper_step1<true, TYPE> {
  static inline int32_t GetRuntimeIndex() {
    return type_index_traits<TYPE>::value;
  }
};

template <typename TYPE>
struct traits_helper_step1<false, TYPE> {
  static inline int32_t GetRuntimeIndex() {
    return traits_helper_step2 < std::is_base_of<ObjectRef, TYPE>::value &&
               object_has_container_type<TYPE>::value,
           TYPE > ::GetRuntimeIndex();
  }
};

template <typename T>
struct traits {
  using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  static int32_t GetRuntimeIndex() {
    return traits_helper_step1<type_index_traits<TYPE>::value != kRuntimeUnknown,
                               TYPE>::GetRuntimeIndex();
  }
};
}  // namespace TypeIndex

}  // namespace runtime
}  // namespace matxscript
