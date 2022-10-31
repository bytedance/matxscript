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

from . import _ffi_api


def structrual_equal(lhs, rhs, assert_mode, map_free_vars):
    """Check structural equality of lhs and rhs.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    assert_mode : bool
        Whether or not throw exception

    map_free_vars : bool
        Whether or not shall we map free vars that does
        not bound to any definitions as equal to each other.

    Returns
    -------

    """
    return _ffi_api.StructuralEqual(lhs, rhs, assert_mode, map_free_vars)


def structural_hash(node, map_free_vars=False):
    """Compute structural hash of node

    The structural hash value is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Normal node: the hash value is defined by its content and type only.
    - Graph node: each graph node will be assigned a unique index ordered by the
      first occurrence during the visit. The hash value of a graph node is
      combined from the hash values of its contents and the index.

    structural_hash is made to be consistent with structural_equal.
    If two nodes are structurally equal to each other,
    then their structural hash (with the same map_free_vars option)
    should be equal to each other as well.

    If the structural hash of two nodes equals to each other,
    then it is highly likely(except for rare hash value collison cases)
    that the two nodes are structurally equal to each other.

    Parameters
    ----------
    node : Object
        The input to be hashed.

    map_free_vars : bool
        If map_free_vars is set to true, we will hash free variables
        by the order of their occurrences. Otherwise, we will hash by
        their in-memory pointer address.

    Return
    ------
    result : int
        The hash result

    See Also
    --------
    structrual_equal
    """
    return _ffi_api.StructuralHash(node, map_free_vars)
