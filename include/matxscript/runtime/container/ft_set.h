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
#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/object.h>

#include <matxscript/runtime/_is_comparable.h>
#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/py_args.h>

#include "_ft_object_base.h"

namespace matxscript {
namespace runtime {

template <typename T>
class MATX_DLL FTSet;

template <typename T>
class MATX_DLL FTSetNode : public FTObjectBaseNode {
 public:
  // data holder
  using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using container_type = ska::flat_hash_set<value_type, SmartHash, SmartEqualTo>;

 public:
  MATXSCRIPT_INLINE_VISIBILITY ~FTSetNode() = default;
  // constructors
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode()
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode(FTSetNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode(const FTSetNode& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode& operator=(FTSetNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode& operator=(const FTSetNode& other) = default;

  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode(IterType first, IterType last)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(first, last) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode(std::initializer_list<value_type> init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(init) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode(const std::vector<value_type>& init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_),
        data_(init.begin(), init.end()) {
  }

 public:
  static const uint64_t type_tag_;
  static const std::type_index std_type_index_;
  static const FTObjectBaseNode::FunctionTable function_table_;
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeFTSet;
  static constexpr const char* _type_key = "FTSet";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FTSetNode, FTObjectBaseNode);

 public:
  container_type data_;
};

template <typename T>
class MATX_DLL FTSet : public FTObjectBase {
 public:
  // data holder
  using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using ContainerType = FTSetNode<value_type>;
  using container_type = typename ContainerType::container_type;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 private:
  // TODO: support custom object eq
  template <class U>
  struct is_comparable_with_value {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    using type = typename std::conditional<
        std::is_same<value_type, U_TYPE>::value ||
            (std::is_arithmetic<value_type>::value && std::is_arithmetic<U_TYPE>::value) ||
            std::is_base_of<Any, value_type>::value || std::is_base_of<Any, U_TYPE>::value ||
            root_type_is_convertible<value_type, U_TYPE>::value,
        std::true_type,
        std::false_type>::type;
    static constexpr bool value = type::value;
  };

 public:
  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;

  // iterators
  MATXSCRIPT_INLINE_VISIBILITY iterator begin() const {
    return MutableImpl().begin();
  }

  MATXSCRIPT_INLINE_VISIBILITY iterator end() const {
    return MutableImpl().end();
  }

  MATXSCRIPT_INLINE_VISIBILITY ~FTSet() noexcept = default;

  // constructors
  MATXSCRIPT_INLINE_VISIBILITY FTSet() : FTObjectBase(make_object<FTSetNode<value_type>>()) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTSet(FTSet&& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSet(const FTSet& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSet& operator=(FTSet&& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTSet& operator=(const FTSet& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  MATXSCRIPT_INLINE_VISIBILITY explicit FTSet(ObjectPtr<Object> n) noexcept
      : FTObjectBase(std::move(n)) {
  }

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTSet(IterType first, IterType last)
      : FTObjectBase(make_object<FTSetNode<value_type>>(first, last)) {
  }

  /*!
   * \brief constructor from initializer FTSet
   * \param init The initializer FTSet
   */
  MATXSCRIPT_INLINE_VISIBILITY FTSet(std::initializer_list<value_type> init)
      : FTObjectBase(make_object<FTSetNode<value_type>>(init)) {
  }

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  MATXSCRIPT_INLINE_VISIBILITY FTSet(const std::vector<value_type>& init)
      : FTObjectBase(make_object<FTSetNode<value_type>>(init)) {
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator==(const FTSet<U>& other) const {
    return this->__eq__(other);
  }
  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator!=(const FTSet<U>& other) const {
    return !operator==(other);
  }

 public:
  template <typename... Args>
  MATXSCRIPT_INLINE_VISIBILITY std::pair<iterator, bool> emplace(Args&&... args) const {
    return MutableImpl().emplace(std::forward<Args>(args)...);
  }

  MATXSCRIPT_INLINE_VISIBILITY void add(value_type item) const {
    MutableImpl().emplace(std::move(item));
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void add(U&& item) const {
    using Converter = GenericValueConverter<value_type>;
    MutableImpl().emplace(Converter()(std::forward<U>(item)));
  }

  MATXSCRIPT_INLINE_VISIBILITY void clear() const {
    MutableImpl().clear();
  }

  MATXSCRIPT_INLINE_VISIBILITY void reserve(int64_t new_size) const {
    if (new_size > 0) {
      MutableImpl().reserve(new_size);
    }
  }

 public:
  // const methods
  MATXSCRIPT_INLINE_VISIBILITY int64_t size() const {
    return MutableImpl().size();
  }

  MATXSCRIPT_INLINE_VISIBILITY int64_t bucket_count() const {
    return MutableImpl().bucket_count();
  }

  MATXSCRIPT_INLINE_VISIBILITY bool empty() const {
    return MutableImpl().empty();
  }

  // method for python
  MATXSCRIPT_INLINE_VISIBILITY Iterator __iter__() const {
    auto iterator_ptr = std::make_shared<iterator>(this->begin());
    auto* iter_c = iterator_ptr.get();
    auto iter_end = this->end();
    auto has_next = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    auto next = [iter_c, iter_end]() -> RTValue {
      RTValue r = value_type(*(*iter_c));
      ++(*iter_c);
      return r;
    };
    auto next_and_checker = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = value_type(*(*iter_c));
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
    return Iterator::MakeGenericIterator(
        *this, std::move(has_next), std::move(next), std::move(next_and_checker));
  }

  MATXSCRIPT_INLINE_VISIBILITY Iterator iter() const {
    return this->__iter__();
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTSet<KEY_U>& o, std::true_type) const {
    return MutableImpl() == o.MutableImpl();
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTSet<KEY_U>& o, std::false_type) const {
    return Iterator::all_items_equal(__iter__(), o.__iter__());
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTSet<KEY_U>& o) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return __eq__(o, std::is_same<KEY_U_TYPE, value_type>{});
  }

  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const Any& o) const {
    if (o.type_code() == TypeIndex::kRuntimeFTSet || o.type_code() == TypeIndex::kRuntimeSet) {
      return Iterator::all_items_equal(__iter__(), Iterator::MakeItemsIterator(o));
    }
    return false;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key, std::true_type) const {
    auto& data_impl = MutableImpl();
    return data_impl.find(key) != data_impl.end();
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key, std::false_type) const {
    return false;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return __contains__(key, typename is_comparable_with_value<KEY_U_TYPE>::type{});
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool contains(KEY_U const& key) const {
    return this->__contains__(key);
  }

  MATXSCRIPT_INLINE_VISIBILITY bool contains(const char* const key) const {
    return this->__contains__(string_view(key));
  }

  MATXSCRIPT_INLINE_VISIBILITY bool contains(const char32_t* const key) const {
    return this->__contains__(unicode_view(key));
  }

 public:
  MATXSCRIPT_INLINE_VISIBILITY void difference_update(PyArgs args) const {
    for (const auto& val : args) {
      if (val.type_code() == TypeIndex::kRuntimeIterator) {
        this->difference_update_iter(val.AsObjectViewNoCheck<Iterator>().data());
      } else {
        this->difference_update_iter(Kernel_Iterable::make(val));
      }
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY FTSet difference(PyArgs args) const {
    FTSet ret(*this);
    ret.difference_update(args);
    return ret;
  }

  MATXSCRIPT_INLINE_VISIBILITY void update(PyArgs args) const {
    for (const auto& val : args) {
      if (val.type_code() == TypeIndex::kRuntimeIterator) {
        this->update_iter(val.AsObjectViewNoCheck<Iterator>().data());
      } else {
        this->update_iter(Kernel_Iterable::make(val));
      }
    }
  }

  // TODO(maxiandi): fix set_union type infer
  MATXSCRIPT_INLINE_VISIBILITY FTSet set_union(PyArgs args) const {
    FTSet ret(*this);
    ret.update(args);
    return ret;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY void discard(const KEY_U& key, std::true_type) const {
    auto& data = MutableImpl();
    auto it = data.find(key);
    if (it != data.end()) {
      data.erase(it);
    }
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY void discard(const KEY_U& key, std::false_type) const {
    SmartHash{}(key);
    return;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY void discard(const KEY_U& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return discard(key, typename is_comparable_with_value<KEY_U_TYPE>::type{});
  }

 private:
  MATXSCRIPT_INLINE_VISIBILITY container_type& MutableImpl() const {
    return static_cast<FTSetNode<value_type>*>(data_.get())->data_;
  }
  MATXSCRIPT_INLINE_VISIBILITY FTSetNode<value_type>* MutableNode() const {
    return static_cast<FTSetNode<value_type>*>(data_.get());
  }

  MATXSCRIPT_INLINE_VISIBILITY void difference_update_iter(const Iterator& iter) const {
    bool has_next = iter.HasNext();
    while (has_next) {
      this->discard(iter.Next(&has_next));
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY void update_iter(const Iterator& iter) const {
    GenericValueConverter<value_type> Converter;
    bool has_next = iter.HasNext();
    while (has_next) {
      this->add(Converter(iter.Next(&has_next)));
    }
  }
};

namespace TypeIndex {
template <typename T>
struct type_index_traits<FTSet<T>> {
  static constexpr int32_t value = kRuntimeFTSet;
};
}  // namespace TypeIndex

// python methods
#define MATXSCRIPT_CHECK_FT_SET_ARGS(FuncName, NumArgs)                               \
  MXCHECK(NumArgs == args.size()) << "[" << DemangleType(typeid(FTSetNode<T>).name()) \
                                  << "::" << #FuncName << "] Expect " << NumArgs      \
                                  << " arguments but get " << args.size()

template <typename T>
const uint64_t FTSetNode<T>::type_tag_ = std::hash<string_view>()(typeid(FTSet<T>).name());
template <typename T>
const std::type_index FTSetNode<T>::std_type_index_ = typeid(FTSet<T>);
template <typename T>
const FTObjectBaseNode::FunctionTable FTSetNode<T>::function_table_ = {
    {"__len__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(__len__, 0);
       return self.AsObjectView<FTSet<T>>().data().size();
     }},
    {"__contains__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(__contains__, 1);
       return self.AsObjectView<FTSet<T>>().data().contains(args[0].template As<RTValue>());
     }},
    {"__eq__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(__eq__, 1);
       return self.AsObjectView<FTSet<T>>().data().__eq__(args[0].template As<RTValue>());
     }},
    {"__iter__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(__iter__, 0);
       return self.AsObjectView<FTSet<T>>().data().iter();
     }},
    {"add",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(add, 1);
       self.AsObjectView<FTSet<T>>().data().add(args[0].template As<RTValue>());
       return None;
     }},
    {"clear",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(clear, 0);
       self.AsObjectView<FTSet<T>>().data().clear();
       return None;
     }},
    {"reserve",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(reserve, 1);
       self.AsObjectView<FTSet<T>>().data().reserve(args[0].As<int64_t>());
       return None;
     }},
    {"bucket_count",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(bucket_count, 0);
       return self.AsObjectView<FTSet<T>>().data().bucket_count();
     }},
    {"difference",
     [](RTView self, PyArgs args) -> RTValue {
       return self.AsObjectView<FTSet<T>>().data().difference(args);
     }},
    {"difference_update",
     [](RTView self, PyArgs args) -> RTValue {
       self.AsObjectView<FTSet<T>>().data().difference_update(args);
       return None;
     }},
    {"union",
     [](RTView self, PyArgs args) -> RTValue {
       return self.AsObjectView<FTSet<T>>().data().set_union(args);
     }},
    {"discard",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_SET_ARGS(discard, 1);
       self.AsObjectView<FTSet<T>>().data().discard(args[0].template As<RTValue>());
       return None;
     }},
    {"update",
     [](RTView self, PyArgs args) -> RTValue {
       self.AsObjectView<FTSet<T>>().data().update(args);
       return None;
     }},
};

#undef MATXSCRIPT_CHECK_FT_SET_ARGS

template <typename value_type>
static inline std::ostream& operator<<(std::ostream& os, FTSet<value_type> const& n) {
  os << '{';
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) {
      os << ", ";
    }
    if (std::is_same<value_type, String>::value) {
      os << "b'" << *it << "'";
    } else if (std::is_same<value_type, Unicode>::value) {
      os << "\'" << *it << "\'";
    } else {
      os << *it;
    }
  }
  os << '}';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
