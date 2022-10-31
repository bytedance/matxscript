# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

class NameRule(object):
    @staticmethod
    def rename_class_method(class_name: str, method_name: str):
        return class_name + "__F_" + method_name

    @staticmethod
    def recover_class_method(class_name: str, unbound_method_name: str):
        prefix = class_name + "__F_"
        assert unbound_method_name.startswith(prefix)
        return unbound_method_name[len(prefix):]

    @staticmethod
    def get_class_init(class_name: str):
        return NameRule.rename_class_method(class_name, "__init__")

    @staticmethod
    def get_class_init_wrapper(class_name: str):
        return NameRule.get_class_init(class_name) + "_wrapper"

    @staticmethod
    def rename_global_method(method_name):
        return "global_" + method_name
