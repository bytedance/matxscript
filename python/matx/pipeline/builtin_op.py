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
import functools


@functools.lru_cache(maxsize=None)
def __get_interpreter_op_impl(opcode: str,
                              py_source_file: str = "",
                              py_source_line: int = -1,
                              py_source_func: str = "",
                              py_source_stmt: str = ""):
    from .ops import InterpreterOp
    return InterpreterOp(opcode=opcode,
                         py_source_file=py_source_file,
                         py_source_line=py_source_line,
                         py_source_func=py_source_func,
                         py_source_stmt=py_source_stmt)


def get_interpreter_op(opcode: str):
    from ._base import current_user_frame
    frame = current_user_frame()
    if frame is None:
        return __get_interpreter_op_impl(opcode=opcode)
    else:
        return __get_interpreter_op_impl(opcode=opcode,
                                         py_source_file=frame.filename,
                                         py_source_line=frame.lineno,
                                         py_source_func=frame.name,
                                         py_source_stmt=frame.line)
