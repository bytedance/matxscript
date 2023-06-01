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

class ExpressionBaseNode:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.range = None

    def to_matx_ir(self, **kwargs):
        raise NotImplementedError("to_matx_ir is not implemented")

    def buffer_regions(self, **kwargs):
        return []

    def reads(self):
        return self.buffer_regions()

    def writes(self):
        return []


class StatementBaseNode:
    def to_matx_ir(self, **kwargs):
        raise NotImplementedError("to_matx_ir is not implemented")

    def reads(self):
        raise NotImplementedError("reads is not implemented")

    def writes(self):
        raise NotImplementedError("writes is not implemented")

    def alocate_buffer(self):
        raise NotImplementedError("alocate buffer is not implemented")
