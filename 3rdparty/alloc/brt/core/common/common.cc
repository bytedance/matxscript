// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0

#include "core/common/common.h"

namespace brt {

// Disable stack for now. TODO add it back
std::vector<std::string> GetStackTrace() {
  return {};
}
}  // namespace brt