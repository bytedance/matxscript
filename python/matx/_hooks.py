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
import multiprocessing
import sys
import json


def _matx_wrap_excepthook(exception_hook):
    """Wrap given excepthook with TVM additional work."""

    def wrapper(exctype, value, trbk):
        """Clean subprocesses when TVM is interrupted."""
        exception_hook(exctype, value, trbk)
        if hasattr(multiprocessing, 'active_children'):
            # pylint: disable=not-callable
            for p in multiprocessing.active_children():
                p.terminate()

    return wrapper


sys.excepthook = _matx_wrap_excepthook(sys.excepthook)


# Enhance default JSONEncoder for MATX support
class _Encoder(json.JSONEncoder):
    def encode(self, o) -> str:
        from .runtime import _ffi_api
        from matx.runtime import List, Dict, Set, Tuple

        if isinstance(o, (List, Dict, Set, Tuple)):
            indent = self.indent
            if indent is None:
                indent = -1
            return _ffi_api.JsonDumps(o, indent, self.ensure_ascii)
        return super().encode(o)

    def default(self, obj):
        from matx.runtime import List, Dict, Tuple

        if isinstance(obj, List):
            return list(obj)
        elif isinstance(obj, Dict):
            return dict(obj)
        elif isinstance(obj, Tuple):
            return tuple(obj)
        else:
            return super().default(obj)


json.JSONEncoder = _Encoder
json._default_encoder = _Encoder(
    skipkeys=False,
    ensure_ascii=True,
    check_circular=True,
    allow_nan=True,
    indent=None,
    separators=None,
    default=None,
)
