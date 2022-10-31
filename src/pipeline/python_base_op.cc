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
#include <matxscript/pipeline/python_base_op.h>

#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(PythonBaseOp).SetThreadSafety(false);

void PythonBaseOp::Init() {
  py_op_name = GetAttr<Unicode>("py_op_name").encode();
  pass_op_name = GetAttr<Unicode>("pass_op_name").encode();
  pass_op_options = GetAttr<Dict>("pass_op_options");

  if (HasAttr("py_callable")) {
    py_callable = GetAttr<NativeFunction>("py_callable");
  }

  // re initialize op name
  name_ = GlobalUniqueIndex::instance()->gen_uniq_name(pass_op_name, name_);

  sub_op_deps.clear();
  auto sub_ops_config = GetAttr<Dict>("sub_op_deps", {});
  for (auto& item : sub_ops_config.items()) {
    auto op_cls = item.first.As<Unicode>().encode();
    auto op_names = item.second.AsObjectRef<List>();
    sub_op_deps.emplace(op_cls, std::vector<String>{});
    for (auto& op_name : op_names) {
      auto op_name_s = op_name.As<Unicode>().encode();
      sub_op_deps[op_cls].push_back(op_name_s);
      auto op_ptr = GetOpImpl(op_cls, op_name_s);
      MXCHECK(op_ptr != nullptr) << "op not found, class: " << op_cls << ", name: " << op_name_s;
    }
  }
}

RTValue PythonBaseOp::Process(PyArgs inputs) const {
  MXCHECK(py_callable) << "[PythonBaseOp] internal error: python callable object is not defined!!!";
  return py_callable(inputs);
}

}  // namespace runtime
}  // namespace matxscript
