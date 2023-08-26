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

from enum import Enum


class FuncReturnKind(Enum):
    VOID = 1,
    SCALAR = 2,
    STATIC_TENSOR = 3,
    DYNAMIC_TENSOR = 4,
    TEMPLATE = 5,

    def is_void(self):
        return self is FuncReturnKind.VOID

    def is_scalar(self):
        return self is FuncReturnKind.SCALAR

    def is_static_tensor(self):
        return self is FuncReturnKind.STATIC_TENSOR

    def is_dynamic_tensor(self):
        return self is FuncReturnKind.DYNAMIC_TENSOR

    def is_template(self):
        return self is FuncReturnKind.TEMPLATE
