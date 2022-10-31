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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * File container
 *****************************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("runtime.File")
    .set_body_typed([](const Unicode& path, const Unicode& mode, const Unicode& encoding) {
      return File(path, mode, encoding);
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.FileHasNext").set_body_typed([](const File& file) {
  return file.HasNext();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.FileReadLineString").set_body_typed([](const File& file) {
  return file.ReadLineString();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.FileReadLineUnicode").set_body_typed([](const File& file) {
  return file.ReadLineUnicode();
});

}  // namespace runtime
}  // namespace matxscript
