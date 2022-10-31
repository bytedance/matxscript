// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/main/Modules/clinic/_randommodule.c.h
 * https://github.com/python/cpython/blob/main/Modules/_randommodule.c
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

#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/py_args.h>

namespace matxscript {
namespace runtime {

double kernel_random_random();
void kernel_random_seed();
void kernel_random_seed_unroll(const Any& n, int64_t version = 2);
void kernel_random_seed(PyArgs args);
Tuple kernel_random_getstate();
void kernel_random_setstate(const Tuple& state);
int64_t kernel_random_getrandbits(int64_t k);
double kernel_random_uniform(double a, double b);
double kernel_random_triangular();
double kernel_random_triangular(PyArgs args);
int64_t kernel_random_randint(int64_t a, int64_t b);
double kernel_random_normalvariate(double mu, double sigma);
double kernel_random_lognormvariate(double mu, double sigma);
double kernel_random_expovariate(double lambd);
double kernel_random_vonmisesvariate(double mu, double kappa);
double kernel_random_gammavariate(double alpha, double beta);
double kernel_random_gauss(double mu, double sigma);
double kernel_random_betavariate(double alpha, double beta);
double kernel_random_paretovariate(double alpha);
double kernel_random_weibullvariate(double alpha, double beta);

}  // namespace runtime
}  // namespace matxscript
