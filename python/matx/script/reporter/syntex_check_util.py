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
# reference from https://docs.python.org/3/library/ast.html
unsupported_syntax_error_table = {
    "Lambda": "Lambda expression is not supported for now, please define the function in outer scope.",
    "Await": "Await is not supported for now.",
    "YieldFrom": "YieldFrom is not supported for now.",
    "Starred": "Starred is not supported for now.",
    "Delete": "\"del\" is not supported for matxscript.",
    "AsyncFor": "AsyncFor is not supported for matxscript",
    "With": "\"with\" statement is not supported for matxscript.",
    "AsyncWith": "AsyncWith is not supported for matxscript",
    "Import": "\"import\" is not supported within matxscript.",
    "ImportFrom": "\"import from\" is not supported within matxscript.",
    "Global": "\"global\" is not supported for matxscript.",
    "Nonlocal": "Nonlocal is not supported for matxscript."}

INVALID_ITERATOR_METHODS = [
    "append", "extend", "add", "update", "discard", "pop", "remove", "reserve", "clear"
]
