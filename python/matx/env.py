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

MATX_DEV_MODE = os.environ.get('MATX_DEV_MODE', '').lower()
MATX_DEV_MODE = MATX_DEV_MODE == '1'

MATX_INFO_COLLECTION = os.environ.get('MATX_INFO_COLLECTION', "1").lower()
MATX_INFO_COLLECTION = MATX_INFO_COLLECTION == "1"

MATX_USER_DIR = os.environ.get('MATX_USER_DIR', os.path.expanduser('~/.matxscript/'))
try:
    os.makedirs(MATX_USER_DIR, exist_ok=True)
except:
    print('[WARNING] User directory created failed: ', MATX_USER_DIR, file=sys.stderr)
