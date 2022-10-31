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

from typing import Optional

from ..context import Span
from ...contrib.statistic import counter


class MATXScriptError(Exception):
    """MATX Script Error"""

    def __init__(self,
                 err_msg: str = 'InternalError',
                 span: Optional[Span] = None,
                 err_type: Exception = None):
        err_type_str = "ScriptError"
        if err_type is not None:
            err_type_str = err_type.__name__
        self.typed_err_msg = '{}: {}'.format(err_type_str, err_msg)
        self.has_context_info = False
        self.span = span

    def __str__(self):
        fn_name = self.span.func_name
        err_context = 'File "{}", line {}, in {}'.format(
            self.span.file_name, self.span.lineno, fn_name
        )
        err_line = self.span.source_code.split('\n')[self.span.lineno - 1]
        err_info = err_context + "\n" + err_line + "\n" + self.typed_err_msg
        counter.set('matx_script_error', err_info)
        counter.set('matx_script_error_counter', 1)
        counter.flush()
        return err_info

    def __repr__(self):
        return self.__str__()
