#  // Copyright 2023 ByteDance Ltd. and/or its affiliates.
#  /*
#   * Licensed to the Apache Software Foundation (ASF) under one
#   * or more contributor license agreements.  See the NOTICE file
#   * distributed with this work for additional information
#   * regarding copyright ownership.  The ASF licenses this file
#   * to you under the Apache License, Version 2.0 (the
#   * "License"); you may not use this file except in compliance
#   * with the License.  You may obtain a copy of the License at
#   *
#   *   http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing,
#   * software distributed under the License is distributed on an
#   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   * KIND, either express or implied.  See the License for the
#   * specific language governing permissions and limitations
#   * under the License.
#   */

import utils as typing_utils
import matx.kernel.symbol.utils as symbol_utils


def calculate_output_axis(arr1_shape, arr2_shape, axis_idx):
    axis1 = arr1_shape[axis_idx]
    axis2 = arr2_shape[axis_idx]

    if axis1 is None and axis2 is None:
        raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                          f"because both axes at f{axis_idx} are None after broadcasting")
    if axis1 is None and axis2 is not None:
        return axis2
    if axis1 is not None and axis2 is None:
        return axis1

    if all([symbol_utils.is_symbol(axis1), symbol_utils.is_symbol(axis2)]):
        if symbol_utils.equals(axis1, axis2):
            return symbol_utils.simplify(axis1)
        else:
            SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                        f"because {axis1} is not equal to {axis2}.")

    if symbol_utils.is_symbol(axis1):
        raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape} "
                          f"because {axis1} is a symbol but {axis2} is not.")
    if symbol_utils.is_symbol(axis2):
        raise SyntaxError(f"{arr1_shape} cannot broadcast with {arr2_shape}"
                          f" because {axis1} is not a symbol but {axis2} is.")

    if axis1 == axis2:
        return axis1

    raise SyntaxError("shapes do not match")


def broadcast_with_scalar(arr1_shape, arr2_shape):
    if typing_utils.is_scalar_shape(arr1_shape) and typing_utils.is_scalar_shape(arr2_shape):
        if len(arr1_shape) > len(arr2_shape):
            return arr1_shape, arr1_shape, arr2_shape
        else:
            return arr2_shape, arr1_shape, arr2_shape
    if typing_utils.is_scalar_shape(arr1_shape):
        padding = max(0, len(arr1_shape) - len(arr2_shape))
        padding_shape = [1] * padding
        new_arr2_shape = padding_shape + arr2_shape
        return new_arr2_shape, arr1_shape, new_arr2_shape
    else:
        padding = max(0, len(arr2_shape) - len(arr1_shape))
        padding_shape = [1] * padding
        new_arr1_shape = padding_shape + arr1_shape
        return new_arr1_shape, new_arr1_shape, arr2_shape


def broadcast(arr1_shape, arr2_shape):
    arr1_shape = list(arr1_shape)
    arr2_shape = list(arr2_shape)
    if typing_utils.is_scalar_shape(arr1_shape) or typing_utils.is_scalar_shape(arr2_shape):
        return broadcast_with_scalar(arr1_shape, arr2_shape)

    arr1_shape.reverse()
    arr2_shape.reverse()
    max_ndim = max(len(arr1_shape), len(arr2_shape))
    for i in range(max_ndim):
        if i >= len(arr1_shape):
            arr1_shape.append(None)
        if i >= len(arr2_shape):
            arr2_shape.append(None)

    output_shape = []
    for i in range(max_ndim):
        output_shape.append(calculate_output_axis(arr1_shape, arr2_shape, i))

    output_shape.reverse()
    arr1_shape.reverse()
    arr2_shape.reverse()
    return output_shape, arr1_shape, arr2_shape
