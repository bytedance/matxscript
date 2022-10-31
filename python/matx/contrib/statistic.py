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
import logging
import logging.config
import os
import sys
from ..env import MATX_INFO_COLLECTION

counter_builder = lambda x: None

try:
    if MATX_INFO_COLLECTION:
        from xstat import Counter
        counter_builder = lambda x: Counter(x)
except:
    MATX_INFO_COLLECTION = False
    pass


class MXCounter:

    def __init__(self, space) -> None:
        try:
            self.space = space
            self.backend = counter_builder(space)
            if MATX_INFO_COLLECTION and self.backend:
                self.enable = True
            else:
                self.enable = False
        except:
            pass

    def set(self, key, value):
        try:
            if self.enable:
                self.backend.message()
                self.backend.set(key, value)
        except:
            pass

    def flush(self):
        try:
            if self.enable:
                self.backend.flush()
        except:
            pass


counter = MXCounter("MATXScript")
counter.flush()
