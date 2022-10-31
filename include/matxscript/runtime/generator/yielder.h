// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of Yielder is inspired by pythran.
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

#include <stdint.h>
namespace matxscript {
namespace runtime {

class Yielder {
 public:
  Yielder() : generator_state__(0) {
  }

  bool operator!=(Yielder const& other) const {
    return generator_state__ != other.generator_state__;
  }
  bool operator==(Yielder const& other) const {
    return generator_state__ == other.generator_state__;
  }

  inline void SetState(int64_t stat) {
    generator_state__ = stat;
  }

  inline int64_t GetState() const {
    return generator_state__;
  }

  int64_t generator_state__;
};

}  // namespace runtime
}  // namespace matxscript
