# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: This file originates from incubator-tvm
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
"""Structured error classes in MATX.

Each error class takes an error message as its input.
See the example sections for for suggested message conventions.
To make the code more readable, we recommended developers to
copy the examples and raise errors with the same message convention.

.. note::

    Please also refer to :ref:`error-handling-guide`.
"""
from ._ffi.base import register_error, TError
import os


@register_error
class InternalError(TError):
    """Internal error in the system.

    Examples
    --------
    .. code :: c++

        // Example code C++
        MXLOG(FATAL) << "InternalError: internal error detail.";

    .. code :: python

        # Example code in python
        raise InternalError("internal error detail")
    """

    def __init__(self, msg):
        # Patch up additional hint message.
        if "MATX hint:" not in msg:
            msg += ("\nMATX hint: You hit an internal error. " +
                    "Please open a thread on nlp/matx to report it.")
        super(InternalError, self).__init__(msg)


register_error("ValueError", ValueError)
register_error("TypeError", TypeError)
register_error("AttributeError", AttributeError)
register_error("KeyError", KeyError)
