// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include "matxscript/runtime/half.h"

namespace matxscript {
namespace runtime {

static_assert(std::is_standard_layout<Half>::value, "c10::Half must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}

}  // namespace runtime
}  // namespace matxscript
