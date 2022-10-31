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
#include <iostream>

using namespace ::matxscript::runtime;

int main(int argc, char* argv[]) {
  std::cout << "Object size: " << sizeof(Object) << std::endl;
  std::cout << "ObjectPtr size: " << sizeof(ObjectPtr<Object>) << std::endl;
  std::cout << "ObjectRef size: " << sizeof(ObjectRef) << std::endl;

  std::cout << "String size: " << sizeof(String) << std::endl;

  std::cout << "Unicode size: " << sizeof(Unicode) << std::endl;

  std::cout << "DataType size: " << sizeof(DataType) << std::endl;

  std::cout << "RTValue size: " << sizeof(RTValue) << std::endl;
  return 0;
}
