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

from typing import Dict, Any

from matx.kernel.kernel_parser import KernelParser


class TemplateFunc:

    def __init__(self, func):
        self.fun_dict: Dict[Any, KernelParser] = {}
        self.func = func

    def get_function(self, args_type_list) -> KernelParser:
        args_types = tuple(args_type_list)
        if args_types not in self.fun_dict:
            p = KernelParser(self.func, args_types)
            p.parse()
            self.fun_dict[args_types] = p
        return self.fun_dict[args_types]
