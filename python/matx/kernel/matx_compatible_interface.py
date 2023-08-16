#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import matx
from typing import Any


class KernelFunction:

    def __init__(self, function_name: str) -> None:
        self.op: matx.native.NativeFunction = matx.make_native_function(function_name)

    def __call__(self, *args: Any) -> Any:
        return self.op(*args)


def get_kernel_func(function_name: str) -> matx.native.NativeFunction:
    return matx.make_native_function(function_name)
