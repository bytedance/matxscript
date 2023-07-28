#  // Copyright 2023 ByteDance Ltd. and/or its affiliates.
#  /*
#   * Licensed to the Apache Software Foundation (ASF) under one
#   * or more contributor license agreements.  See the NOTICE file
#   * distributed with this work for additional information
#   * regarding copyright ownership.  The ASF licenses this file
#   * to you under the Apache License, Version 2.0 (the
#   * "License"); you may not use this file except in compliance
#   * with the License.  You may obtain a copy of the License at
#   *
#   *   http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing,
#   * software distributed under the License is distributed on an
#   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   * KIND, either express or implied.  See the License for the
#   * specific language governing permissions and limitations
#   * under the License.
#   */

import sympy


def is_symbol(x):
    return isinstance(x, sympy.Basic)


def is_symbol_expression(x):
    return isinstance(x, sympy.Expr) and not x.is_symbol


def equals(x, y):
    # https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
    return sympy.simplify(x - y) == 0


def compare(x, y):
    simplified = sympy.simplify(x - y)
    if simplified == 0:
        return 0
    if simplified.is_positive():
        return 1
    return -1


def simplify(x):
    return sympy.simplify(x)


def is_symbol_type(t):
    return t is sympy.Basic
