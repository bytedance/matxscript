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
# pylint: disable=invalid-name
"""Runtime Object api"""
from ._loader import matx_script_api

_CLASS_OBJECT = None


def _set_class_object(object_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = object_class

    def _creator():
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        return obj

    matx_script_api.set_class_object(_creator)


def _register_object(index, cls, callback):
    """register object class"""

    def _creator():
        obj = cls.__new__(cls)
        return obj

    matx_script_api.register_object(index, _creator)
    if callback is not None:
        matx_script_api.register_object_callback(index, callback)


ObjectBase = matx_script_api.ObjectBase

_to_runtime_object = matx_script_api.to_runtime_object
