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

# pylint: disable=wrong-import-position,unused-import
from ._c_ext import matx_script_api
from ._c_ext.object import ObjectBase
from ._c_ext.object import _register_object
from ._c_ext.object import _set_class_object
from ._c_ext.object import _to_runtime_object
from ._c_ext.packed_func import PackedFuncBase
from ._c_ext.packed_func import _get_global_func
from ._c_ext.packed_func import to_packed_func
from ._c_ext.packed_func import _set_class_packed_func
from ._c_ext.packed_func import _set_class_module
from ._c_ext.packed_func import _set_class_object_generic
from ._c_ext.packed_func import _register_input_callback
from ._c_ext.types import _set_class_symbol
from ._c_ext.types import _set_fast_pipeline_object_converter
from ._c_ext.types import void_p_to_runtime

op_kernel_call = matx_script_api.op_kernel_call
