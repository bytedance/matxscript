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

import sys
matx = sys.modules['matx']
from ._base import BatchBaseClass


class ToTorch(object):
    def __call__(self, imgs):
        return imgs.torch()

    def __repr__(self):
        return self.__class__.__name__


class ToGpu(BatchBaseClass):
    def __init__(self, device_id: int = -2):
        super().__init__(device_id=device_id)

    def __call__(self, imgs):
        return [matx.array.from_numpy(img, self.device_str) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + "(device={})".format(self.device_str)
