// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Objects/floatobject.c
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

#include <cstddef>
#include <cstdint>

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
namespace py_builtins {

double float_div(double a, double b);
double float_rem(double vx, double wx);
std::pair<double, double> float_divmod(double vx, double wx);
double float_floor_div(double v, double w);

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
