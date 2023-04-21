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
from .ndarray import *
from ... import ir as _ir


class IfNode(StatementBaseNode):
    def __init__(self, condition, body, orelse, span):
        self.condition = condition
        self.body = body
        self.orelse = orelse
        self.span = span

    def to_matx_ir(self, **kwargs):
        then_body = [s.to_matx_ir() for s in self.body]
        else_body = [s.to_matx_ir() for s in self.orelse]
        return _ir.IfThenElse(self.condition.to_matx_ir(),
                              _ir.SeqStmt(then_body, self.span),
                              _ir.SeqStmt(else_body, self.span),
                              self.span)

    def reads(self):
        body_reads = [r for s in self.body for r in s.reads()]
        orelse_reads = [r for s in self.orelse for r in s.reads()]
        return self.condition.reads() + body_reads + orelse_reads

    def writes(self):
        body_writes = [w for s in self.body for w in s.writes()]
        orelse_writes = [w for s in self.orelse for w in s.writes()]
        return self.condition.writes() + body_writes + orelse_writes
