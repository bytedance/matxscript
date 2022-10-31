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

#include <functional>
#include <tuple>
#include <vector>

#include <matxscript/pipeline/attributes.h>
#include <matxscript/pipeline/node.h>

namespace matxscript {
namespace runtime {

class TXSession;

struct NativeOpKernel {
  virtual const char* ClassName() const = 0;
  virtual void Initialize(const String& resource_path, Attributes attrs) = 0;
  virtual RTValue Process(PyArgs input_args) const = 0;
};

struct NativeOp {
 public:
  NativeOp() = default;
  virtual ~NativeOp() = default;

  const std::string& InstanceName() const {
    return instance_name_;
  }

  const std::string& OpName() const {
    return instance_name_;
  }

  Symbol ComposeSymbol(const std::vector<const Symbol*>& args,
                       const ska::flat_hash_map<std::string, const Symbol*>& kwargs) const;

  Symbol ComposeSymbol(const std::vector<const Symbol*>& args) const {
    ska::flat_hash_map<std::string, const Symbol*> kwargs;
    return ComposeSymbol(args, kwargs);
  }

 protected:
  virtual NodePtr CreateNode() const;

 protected:
  TXSession* session_{nullptr};
  std::string op_name_;
  std::string instance_name_;
  Attributes generic_attrs_;
};

}  // namespace runtime
}  // namespace matxscript
