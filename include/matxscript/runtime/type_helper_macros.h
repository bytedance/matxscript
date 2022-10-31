// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#pragma once

#include <sstream>

#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

namespace internal {
namespace {
template <typename T>
struct TypeAsConverterNoCheck {
 private:
  using TO_TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  static_assert(!std::is_same<TO_TYPE, Any>::value, "TypeAsHelper return 'Any' type is disabled");
  static constexpr bool TO_TYPE_IS_OBJECT_REF = std::is_base_of<ObjectRef, TO_TYPE>::value;

  template <typename U>
  static MATXSCRIPT_ALWAYS_INLINE TO_TYPE run(U&& v, std::integral_constant<bool, true>) {
    return v.template MoveToObjectRefNoCheck<TO_TYPE>();
  }

  template <typename U>
  static MATXSCRIPT_ALWAYS_INLINE TO_TYPE run(U&& v, std::integral_constant<bool, false>) {
    return v.template AsNoCheck<TO_TYPE>();
  }

 public:
  template <typename U>
  static MATXSCRIPT_ALWAYS_INLINE TO_TYPE run(U&& v) {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    static constexpr bool move_mode = std::is_rvalue_reference<decltype(v)>::value &&
                                      std::is_same<U_TYPE, RTValue>::value && TO_TYPE_IS_OBJECT_REF;
    return run(std::forward<U>(v), std::integral_constant<bool, move_mode>{});
  }
};

template <>
struct TypeAsConverterNoCheck<String> {
  template <typename U>
  static MATXSCRIPT_ALWAYS_INLINE String run(U&& v) {
    return v.template AsNoCheck<String>();
  }
  static MATXSCRIPT_ALWAYS_INLINE String run(RTValue&& v) {
    return v.MoveToBytesNoCheck();
  }
};

template <>
struct TypeAsConverterNoCheck<Unicode> {
  template <typename U>
  static MATXSCRIPT_ALWAYS_INLINE Unicode run(U&& v) {
    return v.template AsNoCheck<Unicode>();
  }
  static MATXSCRIPT_ALWAYS_INLINE Unicode run(RTValue&& v) {
    return v.MoveToUnicodeNoCheck();
  }
};
}  // namespace

template <typename T>
struct TypeAsHelper {
  using TO_TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  static_assert(!std::is_same<TO_TYPE, Any>::value, "TypeAsHelper return 'Any' type is disabled");

  template <class From, typename = typename std::enable_if<is_runtime_value<From>::value>::type>
  static MATXSCRIPT_ALWAYS_INLINE TO_TYPE
  run(From&& v, const char* file, int line, const char* prefix, const char* expect) {
    using FROM_TYPE = typename std::remove_cv<typename std::remove_reference<From>::type>::type;
    static_assert(!std::is_same<FROM_TYPE, TO_TYPE>::value,
                  "TypeAsHelper from type is same as to type");
    if (std::is_base_of<Any, TO_TYPE>::value) {
      return v.template As<TO_TYPE>();
    }
    bool state = v.template Is<TO_TYPE>();
    if (std::is_same<double, TO_TYPE>::value) {
      state |= v.template Is<int64_t>();
    }
    if (!state) {
      auto v_name = v.type_name();
      if (v_name == "Unicode" || v_name == "unicode_view") {
        v_name = "py::str";
      } else if (v_name == "String" || v_name == "string_view") {
        v_name = "py::bytes";
      }
      std::string message;
      if (prefix) {
        message.append(prefix);
      }
      message.append(expect);
      message.append(", but get '").append(v_name.data(), v_name.size()).append("'");
      throw TypeError(details::FormatLineMessage(file, line, "TypeError", message));
    }
    return TypeAsConverterNoCheck<TO_TYPE>::run(std::forward<From>(v));
  }
};

}  // namespace internal

}  // namespace runtime
}  // namespace matxscript

#define MATXSCRIPT_TYPE_AS(o, TYPE)                         \
  ::matxscript::runtime::internal::TypeAsHelper<TYPE>::run( \
      (o), __FILE__, __LINE__, nullptr, "expect '" #o "' is '" #TYPE "' type")

// this is for python only
#define MATXSCRIPT_TYPE_AS_WITH_PY_INFO(o, TYPE, PY_INFO)   \
  ::matxscript::runtime::internal::TypeAsHelper<TYPE>::run( \
      (o), __FILE__, __LINE__, (PY_INFO), "expect '" #o "' is '" #TYPE "' type")
