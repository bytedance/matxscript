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
import warnings
from .._ffi.base import string_types
from .symbol import Variable as Var
from .ops import make_op_creator_function

_warm_up_op_creator = make_op_creator_function("WarmUpV3Op")


class WarmUp(object):

    def __init__(self, feed_dict=None, static_batch_fields=None):
        if static_batch_fields is None:
            static_batch_fields = []
        if feed_dict is None:
            feed_dict = dict()
        warm_op = _warm_up_op_creator(feed_dict=feed_dict,
                                      static_batch_fields=static_batch_fields)

        def warmup_func(batch):
            return warm_op(batch)

        batch_size = Var("batch_size", 1)
        from . import Trace
        self._warm_up_module = Trace(warmup_func, batch_size)

    def Gen(self, batch_size):
        warnings.warn("The function WarmUp.Gen is deprecated.", DeprecationWarning)
        return self.gen(batch_size)

    def gen(self, batch_size):
        """Gen a module feed dict by specified batch_size

        Parameters
        ----------
        batch_size : int
            Expected batch size

        Returns
        -------
        feed_dict : dict(str, OpData)
            a sample feed dict

        """
        if sys.version_info[0] == 3:
            assert isinstance(batch_size, int)
        else:
            assert isinstance(batch_size, int) or isinstance(batch_size, long)
        warm_up_ret = self._warm_up_module.run(batch_size=batch_size)
        return warm_up_ret

    def Save(self, folder):
        warnings.warn("The function WarmUp.Save is deprecated.", DeprecationWarning)
        return self.save(folder)

    def save(self, folder):
        """Save warm up data to folder

        Parameters
        ----------
        folder : str
            warmup data folder

        Returns
        -------

        """
        assert isinstance(folder, string_types)
        name = "warmup_data.json"
        self._warm_up_module.save(folder, name)


def make(feed_dict, static_batch_fields=None):
    """Make a WarmUp object

    Examples
    --------
    .. code-block:: python

        import matx
        from matx import warmup

        # all items in feed_dict batch must be 1
        wp = warmup.make(feed_dict={"raw/texts": ["hello"]})

        # gen a test
        batch_data = wp.Gen(batch_size=2)
        print(batch_data) # {"raw/texts": ["hello", "hello"]}

        # save
        wp.save("./") # save to warmup_data.json


    Parameters
    ----------
    feed_dict : dict
        a sample feed dict with batch size equal to 1

    static_batch_fields : list(str)
        Parameter names that require constant batch, Default None

    Returns
    -------
    wp : WarmUp
        warm up helper object

    """
    return WarmUp(feed_dict, static_batch_fields)


__all__ = ["WarmUp", "make"]
