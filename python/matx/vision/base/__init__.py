# -*- coding:utf-8 -*-

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

import os
import sys
from ._libinfo import find_lib_path
from ._dso_loader import load_bundled_lib

LIB = None
try:
    LIB = load_bundled_lib("libbyted_vision_cpu_ops")
except IOError as e:
    print(
        "[WARN][BytedVision] Can't found CUDA HOME. When use GPU Runtime, please set CUDA HOME to LD_LIBRARY_PATH ! ERROR: ",
        e,
        file=sys.stderr)
except Exception as e:
    print("[ERROR][BytedVision] Occur Error when load byted_vision cuda ops: ", e, file=sys.stderr)

CUDA_LIB = None
BYTED_VISION_SYNC = os.environ.get('BYTED_VISION_SYNC', '')
BYTED_VISION_SYNC = BYTED_VISION_SYNC == '1'

try:
    CUDA_LIB = load_bundled_lib("libbyted_vision_cuda_ops")
except IOError as e:
    print(
        "[WARN][BytedVision] Can't found CUDA HOME. When use GPU Runtime, please set CUDA HOME to LD_LIBRARY_PATH ! ERROR: ",
        e,
        file=sys.stderr)
except Exception as e:
    print("[ERROR][BytedVision] Occur Error when load byted_vision cuda ops: ", e, file=sys.stderr)
