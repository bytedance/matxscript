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
#include <matxscript/pipeline/py_torch_infer_op.h>

#include <matxscript/pipeline/tx_session.h>

#include "userdata_mutator.h"

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(PyTorchInferOp).SetThreadSafety(false);

void PyTorchInferOp::Init() {
  auto impl_data = GetAttr<RTValue>("impl");
  UserDataMutator::Mutate(&impl_data, this);
  auto ud_ref = impl_data.AsObjectRef<UserDataRef>();
  impl_ = check_get_op_kernel(ud_ref);
  sub_ops_ = {impl_};
}

RTValue PyTorchInferOp::Process(PyArgs inputs) const {
  return impl_->Process(inputs);
}

}  // namespace runtime
}  // namespace matxscript
