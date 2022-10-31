// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
 * - implement some tuple methods
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
#include <matxscript/runtime/container/tuple_private.h>

#include <matxscript/runtime/memory.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Tuple container
 *****************************************************************************/

MATXSCRIPT_REGISTER_OBJECT_TYPE(TupleNode);

ObjectPtr<TupleNode> TupleNode::MakeNones(size_t n) {
  auto output_node = make_inplace_array_object<TupleNode, TupleNode::value_type>(n);
  output_node->size = 0;
  for (size_t i = 0; i < n; ++i) {
    output_node->EmplaceInit(i, nullptr);
    // Only increment size after the initialization succeeds
    output_node->size++;
  }
  return output_node;
}

}  // namespace runtime
}  // namespace matxscript
