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


def func(compiling_obj, *args, **kwargs):
    from .kernel_parser import KernelParser
    from .compile_linalg import compile_linalg
    p = KernelParser(compiling_obj)
    p.parse()
    return compile_linalg(p)


def template(compiling_obj, *args, **kwargs):
    from ._template import TemplateFunc
    from .func_registery import TEMPLATE_REGISTRY

    if compiling_obj not in TEMPLATE_REGISTRY:
        TEMPLATE_REGISTRY[compiling_obj] = TemplateFunc(compiling_obj)
    return compiling_obj
