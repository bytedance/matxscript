// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#pragma once

#include <matxscript/runtime/container/kwargs_ref.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {
namespace list_details {

inline void trait_sort_kwargs(const PyArgs& args, RTValue** key_func, bool* reverse) {
  *key_func = nullptr;
  *reverse = false;
  auto num_args = args.size();
  if (num_args != 1 || args[num_args - 1].type_code() != TypeIndex::kRuntimeKwargs) {
    THROW_PY_TypeError("sort() takes no positional arguments");
  }
  auto kwargs = args[num_args - 1].AsObjectRefNoCheck<Kwargs>();
  int64_t num_kwargs = kwargs.size();

  if (kwargs.contains("key")) {
    *key_func = &(kwargs["key"]);
    --num_kwargs;
  }
  if (kwargs.contains("reverse")) {
    --num_kwargs;
    auto reverse_any = kwargs["reverse"];
    if (reverse_any.Is<int64_t>()) {
      *reverse = reverse_any.AsNoCheck<int64_t>();
    } else {
      THROW_PY_TypeError("an integer is required (got type ", reverse_any.type_name(), ")");
    }
  }
  if (num_kwargs != 0) {
    static string_view arg_names[2]{"key", "reverse"};
    THROW_PY_TypeError(
        "list.sort() got an unexpected keyword argument '", kwargs.diff(arg_names, 2), "'");
  }
}

}  // namespace list_details
}  // namespace runtime
}  // namespace matxscript
