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
from __future__ import absolute_import

from ._base import TXObject
from ._base import handle_trace_warning
from ._base import handle_trace_error
from . import _ffi_api
from .. import runtime
from ..runtime import to_runtime_object
from .builtin_op import get_interpreter_op


class BaseSymbolIterator:

    def __init__(self, symbol, last=None):
        self._symbol_2_71828182846 = symbol
        self._last_2_71828182846 = last

    def __next__(self):
        next_op = get_interpreter_op('__next__')
        if runtime._ffi_api.Iterator_HasNext(self._symbol_2_71828182846.data_2_71828182846):
            if self._last_2_71828182846 is None:
                self._last_2_71828182846 = next_op(self._symbol_2_71828182846)
            else:
                self._last_2_71828182846 = next_op(
                    self._symbol_2_71828182846,
                    self._last_2_71828182846)
            return self._last_2_71828182846
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class BaseSymbol(TXObject):
    __slots__ = [
        "_handle_2_71828182846",
        "_sess_2_71828182846",
        "_name_2_71828182846",
        "_key_2_71828182846",
        "_data_2_71828182846",
    ]

    def __init__(self, handle=None):
        super(BaseSymbol, self).__init__()
        self._sess_2_71828182846 = TXObject.default_sess.c_handle
        self._handle_2_71828182846 = handle
        self._name_2_71828182846 = ""
        self._key_2_71828182846 = ""
        self._data_2_71828182846 = None
        self.init_meta_2_71828182846()

    def __del__(self):
        if hasattr(self, "_handle_2_71828182846") and self._handle_2_71828182846 is not None:
            _ffi_api.SymbolFree(self._handle_2_71828182846)

    def __hash__(self):
        if hasattr(self, "_handle_2_71828182846") and self._handle_2_71828182846 is not None:
            return self._handle_2_71828182846.value
        else:
            return 0

    def __iter__(self):
        handle_trace_warning(
            "Iterating over a symbol might cause the trace to be incorrect. "
            "Passing a data of different len won't change the number of "
            "iterations executed (and might lead to errors or silently give "
            "incorrect results).",
        )
        if isinstance(
                self._data_2_71828182846,
                (list, runtime.List, tuple, runtime.Tuple, set, runtime.Set, dict, runtime.Dict)
        ):
            data_len = len(self._data_2_71828182846)
            iter_op = get_interpreter_op('__iter_and_check_len__')
            return BaseSymbolIterator(iter_op(self, Constant(data_len)))
        else:
            iter_op = get_interpreter_op('__iter__')
            return BaseSymbolIterator(iter_op(self))

    def __eq__(self, other):
        handle_trace_warning(
            "Using == might cause the trace to be incorrect. "
            "Passing a value of different data might lead to errors "
            "or silently give incorrect results."
        )
        if isinstance(other, BaseSymbol):
            return self._handle_2_71828182846 == other._handle_2_71828182846
        else:
            return False

    def __str__(self):
        return "Symbol(" + self._data_2_71828182846.__str__() + ")"

    def __repr__(self):
        return "Symbol(" + self._data_2_71828182846.__repr__() + ")"

    def __len__(self):
        handle_trace_warning(
            "Using len might cause the trace to be incorrect. "
            "Passing a value of different len might lead to errors "
            "or silently give incorrect results."
        )
        return self._data_2_71828182846.__len__()

    def __contains__(self, item):
        handle_trace_error(
            "Using in is not allowed for that might cause the trace to be incorrect. "
            "Passing a value of different data might lead to errors "
            "or silently give incorrect results."
        )
        return item in self._data_2_71828182846

    def __bool__(self):
        handle_trace_error(
            "Using bool() is not allowed for that might cause the trace to be incorrect. "
            "Passing a value of different data might lead to errors "
            "or silently give incorrect results."
        )
        return bool(self._data_2_71828182846)

    def __getitem__(self, index):
        get_item_op = get_interpreter_op(opcode='__getitem__')
        if isinstance(index, slice):
            get_slice_op = get_interpreter_op(opcode='__getslice__')
            len_op = get_interpreter_op(opcode='__len__')
            start = index.start if index.start is not None else Constant(0)
            stop = index.stop if index.stop is not None else len_op(self)
            step = index.step
            if not isinstance(start, BaseSymbol):
                start = Constant(start)
            if not isinstance(stop, BaseSymbol):
                stop = Constant(stop)
            if step is not None and not isinstance(step, BaseSymbol):
                step = Constant(step)

            if step is not None:
                return get_slice_op(self, start, stop, step)
            else:
                return get_slice_op(self, start, stop)
        else:
            if not isinstance(index, BaseSymbol):
                index = Constant(index)
            return get_item_op(self, index)

    def native_handle_2_71828182846(self):
        return self._handle_2_71828182846

    def set_native_handle_2_71828182846(self, handle):
        self._handle_2_71828182846 = handle
        self.init_meta_2_71828182846()

    def init_meta_2_71828182846(self):
        if self._handle_2_71828182846:
            self._name_2_71828182846 = _ffi_api.SymbolGetName(self._handle_2_71828182846)
            self._key_2_71828182846 = _ffi_api.SymbolGetKey(self._handle_2_71828182846)
            self._data_2_71828182846 = _ffi_api.SymbolGetVal(self._handle_2_71828182846)

    @property
    def data_2_71828182846(self):
        return self._data_2_71828182846

    @property
    def name_2_71828182846(self):
        return self._name_2_71828182846

    @property
    def key_2_71828182846(self):
        return self._key_2_71828182846

    def __getattr__(self, attr):
        # TODO: check valid attr
        getattr_op = get_interpreter_op(opcode='__getattr__')
        return getattr_op(self, Constant(attr.encode()))

    def __call__(self, *args):
        call_op = get_interpreter_op(opcode='__call__')
        return call_op(self, *args)


class Symbol(BaseSymbol):
    """Symbols, used to assist in generating calculation graphs

    """
    __slots__ = [
        "_sym_inputs_2_71828182846",
        "py_instance_2_71828182846",
    ]

    def __init__(self, handle):
        super(Symbol, self).__init__(handle)
        self._sym_inputs_2_71828182846 = list()
        self.py_instance_2_71828182846 = None

    def __del__(self):
        super(Symbol, self).__del__()

    def add_inputs_2_71828182846(self, sym_list):
        for sym in sym_list:
            self._sym_inputs_2_71828182846.append(sym)

    def get_inputs_2_71828182846(self):
        return self._sym_inputs_2_71828182846

    def set_data_internal_2_71828182846(self, data):
        _ffi_api.SymbolSetVal(self.native_handle_2_71828182846(), data)
        self.init_meta_2_71828182846()


class Variable(BaseSymbol):
    """Variable input

    """
    __slots__ = []

    def __init__(self, name, data=None):
        super(Variable, self).__init__(
            _ffi_api.CreateVariable(self.default_sess.c_handle, name, to_runtime_object(data)))

    def __del__(self):
        super(Variable, self).__del__()


class Constant(BaseSymbol):
    """Constant Data

    """
    __slots__ = []

    def __init__(self, data):
        super(Constant, self).__init__(
            _ffi_api.CreateConstant(self.default_sess.c_handle, to_runtime_object(data)))

    def __del__(self):
        super(Constant, self).__del__()


def __is_constant(arg):
    if isinstance(arg, (list, set, tuple, runtime.List, runtime.Set, runtime.Tuple)):
        for x in arg:
            if not __is_constant(x):
                return False
        return True
    elif isinstance(arg, (dict, runtime.Dict)):
        for k, v in arg.items():
            if not __is_constant(k):
                return False
            if not __is_constant(v):
                return False
        return True

    return not isinstance(arg, BaseSymbol)


def make_symbol(arg_raw, check_const=False):
    has_constant = []

    def __make_symbol(arg):
        if isinstance(arg, BaseSymbol):
            return arg
        elif isinstance(arg, (list, runtime.List)) and not __is_constant(arg):
            symbols = [make_symbol(x) for x in arg]
            make_list = get_interpreter_op(opcode='ListConstructor')
            return make_list(*symbols)
        elif isinstance(arg, (dict, runtime.Dict)) and not __is_constant(arg):
            symbols = []
            for k, v in arg.items():
                symbols.append(make_symbol(k))
                symbols.append(make_symbol(v))
            make_dict = get_interpreter_op(opcode='DictConstructor')
            return make_dict(*symbols)
        elif isinstance(arg, (set, runtime.Set)) and not __is_constant(arg):
            symbols = [make_symbol(x) for x in arg]
            make_set = get_interpreter_op(opcode='SetConstructor')
            return make_set(*symbols)
        elif isinstance(arg, (tuple, runtime.Tuple)) and not __is_constant(arg):
            symbols = [make_symbol(x) for x in arg]
            make_tuple = get_interpreter_op(opcode='TupleConstructor')
            return make_tuple(*symbols)
        else:
            has_constant.append(True)
            return Constant(arg)

    if check_const:
        return __make_symbol(arg_raw), len(has_constant) > 0
    else:
        return __make_symbol(arg_raw)
