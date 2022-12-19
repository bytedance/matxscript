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
import inspect
import sys

from ..native import make_native_object
from ..native import make_native_function
from ..native import call_native_function

from . import _ffi_api
from . import op as _ir_op
from . import adt as _ir_adt
from . import expr as _expr
from . import type as _type
from . import InitializerList
from . import const

_module_name_ = "matx"


class Builtin2Op(object):
    registrations = dict()

    @staticmethod
    def lookup(name_or_type):
        if name_or_type in Builtin2Op.registrations:
            return Builtin2Op.registrations[name_or_type]
        if not isinstance(name_or_type, (str, bytes, bytearray)):
            mod = inspect.getmodule(name_or_type)
            if mod is not None:
                module_name = mod.__name__
                absolute_name = module_name + "." + name_or_type.__name__
                if absolute_name in Builtin2Op.registrations:
                    return Builtin2Op.registrations[absolute_name]
        return None

    @staticmethod
    def lookup_with_dynamic_type(name_or_type, expect_ret_type):
        symbol = Builtin2Op.lookup(name_or_type)
        if (expect_ret_type is not None
                and expect_ret_type.is_full_typed()
                and isinstance(symbol, _ir_adt.Constructor)):
            if (isinstance(symbol.checked_type, _type.ListType)
                    and isinstance(expect_ret_type, _type.ListType)):
                symbol = _ir_adt.Constructor("FTList",
                                             inputs=[expect_ret_type.item_type],
                                             ret_type=expect_ret_type)
            elif (isinstance(symbol.checked_type, _type.SetType)
                  and isinstance(expect_ret_type, _type.SetType)):
                symbol = _ir_adt.Constructor("FTSet",
                                             inputs=[expect_ret_type.item_type],
                                             ret_type=expect_ret_type)
            elif (isinstance(symbol.checked_type, _type.DictType)
                  and isinstance(expect_ret_type, _type.DictType)):
                symbol = _ir_adt.Constructor(
                    "FTDict",
                    inputs=[expect_ret_type.key_type,
                            expect_ret_type.value_type],
                    ret_type=expect_ret_type)
        return symbol


def _register_op(builtin_func, _op):
    Builtin2Op.registrations[builtin_func] = _op


def _register_container_cons(name_hint, ret_type):
    assert isinstance(name_hint, str) and name_hint.startswith("{}.".format(_module_name_))
    Builtin2Op.registrations[name_hint] = _ir_adt.Constructor(
        name_hint.split('.')[-1], ret_type=ret_type)


def _register_builtin_container_cons(name_hint, matx_name_hint, ret_type):
    Builtin2Op.registrations[name_hint] = _ir_adt.Constructor(matx_name_hint, ret_type=ret_type)


def _register_pod_cons(py_name_hint, cc_name_hint, ret_type):
    pod_cons = _ir_adt.Constructor(cc_name_hint, ret_type=ret_type)

    def fused_pod_cons(span, *args):
        if py_name_hint == "int" and len(args) == 1:
            fused_expr = _ffi_api.TryNDArrayItemAsInt64(args[0])
            if fused_expr is not None:
                return fused_expr
        # if py_name_hint == "bool" and len(args) == 1:
        #     fused_expr = _ffi_api.TryNDArrayItemAsInt64(args[0])
        #     if fused_expr is not None:
        #         return pod_cons(span, fused_expr)
        elif py_name_hint == "float" and len(args) == 1:
            fused_expr = _ffi_api.TryNDArrayItemAsDouble(args[0])
            if fused_expr is not None:
                return fused_expr
        return pod_cons(span, *args)

    fused_pod_cons.is_constructor = True
    fused_pod_cons.checked_type = ret_type
    Builtin2Op.registrations[py_name_hint] = fused_pod_cons


def _register_object_builtin_op(method):
    base_func_name = "object_" + method
    base_key_name = "ir.object_" + method
    Builtin2Op.registrations[base_key_name] = getattr(_ir_op, base_func_name)


def _register_str_builtin_op(method):
    base_func_name = "str_" + method
    base_key_name = "ir.str_" + method
    Builtin2Op.registrations[base_key_name] = getattr(_ir_op, base_func_name)


def _register_python_builtin(method, _method=None):
    if _method is None:
        base_func_name = "object_" + method
    else:
        base_func_name = _method
    Builtin2Op.registrations[method] = getattr(_ir_op, base_func_name)


###############################################################################
# Python builtin functions
###############################################################################
_register_python_builtin("len")
_register_python_builtin("open", "builtins_open")
_register_python_builtin("ord", "builtins_ord")
_register_python_builtin("chr", "builtins_chr")
_register_python_builtin("enumerate", "builtins_enumerate")
_register_python_builtin("zip", "builtins_zip")
_register_python_builtin("isinstance", "builtins_isinstance")
_register_python_builtin("sorted", "builtins_sorted")

###############################################################################
# Any object.method
###############################################################################
_register_object_builtin_op("append")
_register_object_builtin_op("add")
_register_object_builtin_op("extend")
_register_object_builtin_op("contains")
_register_object_builtin_op("clear")
_register_object_builtin_op("get_item")
_register_object_builtin_op("set_item")
_register_object_builtin_op("find")
_register_object_builtin_op("lower")
_register_object_builtin_op("upper")
_register_object_builtin_op("isdigit")
_register_object_builtin_op("isalpha")
_register_object_builtin_op("encode")
_register_object_builtin_op("decode")
_register_object_builtin_op("get_slice")
_register_object_builtin_op("set_slice")
_register_object_builtin_op("split")
_register_object_builtin_op("join")
_register_object_builtin_op("startswith")
_register_object_builtin_op("endswith")
_register_object_builtin_op("rstrip")
_register_object_builtin_op("lstrip")
_register_object_builtin_op("strip")
_register_object_builtin_op("count")
_register_object_builtin_op("pop")
_register_object_builtin_op("remove")
_register_object_builtin_op("format")
_register_object_builtin_op("insert")
_register_object_builtin_op("index")

_register_str_builtin_op("split_ft")

_register_object_builtin_op("readline")
_register_object_builtin_op("readlines")
_register_object_builtin_op("reserve")
_register_object_builtin_op("capacity")
_register_object_builtin_op("bucket_count")
_register_object_builtin_op("update")
_register_object_builtin_op("prefix_search")
_register_object_builtin_op("prefix_search_all")
_register_object_builtin_op("save")
_register_object_builtin_op("load")
_register_object_builtin_op("replace")
_register_object_builtin_op("match")
_register_object_builtin_op("to_list")
_register_object_builtin_op("tolist")
_register_object_builtin_op("is_contiguous")
_register_object_builtin_op("keys")
_register_object_builtin_op("values")
_register_object_builtin_op("items")
_register_object_builtin_op("get")
_register_object_builtin_op("shape")
_register_object_builtin_op("dtype")
_register_object_builtin_op("dim")
_register_object_builtin_op("device")
_register_object_builtin_op("close")
_register_object_builtin_op("read")
_register_object_builtin_op("difference")
_register_object_builtin_op("difference_update")
_register_object_builtin_op("discard")
_register_object_builtin_op("transpose")
_register_object_builtin_op("as_type")
_register_object_builtin_op("reverse")
_register_object_builtin_op("union")
_register_object_builtin_op("sort")
_register_object_builtin_op("contiguous")
_register_object_builtin_op("reshape")
_register_object_builtin_op("squeeze")
_register_object_builtin_op("unsqueeze")

###############################################################################
# Python builtin modules
###############################################################################
_register_python_builtin("unicodedata.normalize", "unicodedata_normalize")
_register_python_builtin("unicodedata.category", "unicodedata_category")
_register_python_builtin("json.loads", "json_loads")
_register_python_builtin("json.load", "json_load")
_register_python_builtin("json.dumps", "json_dumps")
_register_python_builtin("time.time", "time_time")
_register_python_builtin("os.getenv", "os_getenv")
_register_python_builtin("base64.b64encode", "base64_b64encode")
_register_python_builtin("base64.b64decode", "base64_b64decode")

# random
_register_python_builtin("random.random", "random_random")
_register_python_builtin("random.seed", "random_seed")
_register_python_builtin("random.getstate", "random_getstate")
_register_python_builtin("random.setstate", "random_setstate")
_register_python_builtin("random.getrandbits", "random_getrandbits")
_register_python_builtin("random.uniform", "random_uniform")
_register_python_builtin("random.triangular", "random_triangular")
_register_python_builtin("random.randint", "random_randint")
_register_python_builtin("random.normalvariate", "random_normalvariate")
_register_python_builtin("random.lognormvariate", "random_lognormvariate")
_register_python_builtin("random.expovariate", "random_expovariate")
_register_python_builtin("random.vonmisesvariate", "random_vonmisesvariate")
_register_python_builtin("random.gammavariate", "random_gammavariate")
_register_python_builtin("random.gauss", "random_gauss")
_register_python_builtin("random.betavariate", "random_betavariate")
_register_python_builtin("random.paretovariate", "random_paretovariate")
_register_python_builtin("random.weibullvariate", "random_weibullvariate")

# python math
_register_python_builtin("min", "min")
_register_python_builtin("max", "max")
_register_python_builtin("math.exp", "exp")
_register_python_builtin("math.exp2", "exp2")
_register_python_builtin("math.exp10", "exp10")
_register_python_builtin("math.erf", "erf")
_register_python_builtin("math.tanh", "tanh")
_register_python_builtin("math.sigmoid", "sigmoid")
_register_python_builtin("math.log", "log")
_register_python_builtin("math.log2", "log2")
_register_python_builtin("math.log10", "log10")
_register_python_builtin("math.log1p", "log1p")
_register_python_builtin("math.tan", "tan")
_register_python_builtin("math.cos", "cos")
_register_python_builtin("math.cosh", "cosh")
_register_python_builtin("math.acos", "acos")
_register_python_builtin("math.acosh", "acosh")
_register_python_builtin("math.sin", "sin")
_register_python_builtin("math.sinh", "sinh")
_register_python_builtin("math.asin", "asin")
_register_python_builtin("math.asinh", "asinh")
_register_python_builtin("math.atan", "atan")
_register_python_builtin("math.atanh", "atanh")
_register_python_builtin("math.atan2", "atan2")
_register_python_builtin("math.sqrt", "sqrt")
_register_python_builtin("math.rsqrt", "rsqrt")
_register_python_builtin("math.floor", "floor")
_register_python_builtin("math.ceil", "ceil")
_register_python_builtin("math.trunc", "trunc")
_register_python_builtin("abs", "abs")
_register_python_builtin("math.round", "round")
_register_python_builtin("math.nearbyint", "nearbyint")
_register_python_builtin("math.nextafter", "nextafter")
_register_python_builtin("math.hypot", "hypot")
_register_python_builtin("math.copysign", "copysign")
_register_python_builtin("math.ldexp", "ldexp")
_register_python_builtin("math.isnan", "isnan")
_register_python_builtin("math.isfinite", "isfinite")
_register_python_builtin("math.isinf", "isinf")
_register_python_builtin("math.pow", "power")
_register_python_builtin("math.fmod", "fmod")
_register_python_builtin("print", "builtins_print")

###############################################################################
# runtime extension
###############################################################################

# ndarray
_register_python_builtin("{}.runtime.ndarray.add".format("matx"), "nd_module_add")
_register_python_builtin("{}.runtime.ndarray.sub".format("matx"), "nd_module_sub")
_register_python_builtin("{}.runtime.ndarray.div".format("matx"), "nd_module_div")
_register_python_builtin("{}.runtime.ndarray.mul".format("matx"), "nd_module_mul")
_register_python_builtin("{}.runtime.ndarray.rand".format("matx"), "nd_module_rand")
_register_python_builtin("{}.runtime.ndarray.concatenate".format("matx"), "nd_module_concatenate")
_register_python_builtin("{}.runtime.ndarray.stack".format("matx"), "nd_module_stack")

# TODO: support heapq
_register_python_builtin("{}.list_sort".format("matx"), "list_module_sort")
_register_python_builtin(
    "{}.runtime._container._list.heapify".format("matx"),
    "list_module_heapify")
_register_python_builtin(
    "{}.runtime._container._list.heap_replace".format("matx"),
    "list_module_heap_replace")
_register_python_builtin(
    "{}.runtime._container._list.heap_pushpop".format("matx"),
    "list_module_heap_pushpop")
_register_python_builtin(
    "{}.runtime._container._list.nth_element".format("matx"),
    "list_module_nth_element")

# cuda_stream
_register_python_builtin(
    "{}.runtime.cuda_stream.default_stream".format("matx"),
    "cuda_module_default_stream")
_register_python_builtin(
    "{}.runtime.cuda_stream.create_stream".format("matx"),
    "cuda_module_create_stream")
_register_python_builtin(
    "{}.runtime.cuda_stream.stream_sync".format("matx"),
    "cuda_module_stream_sync")


def _str2ir(span, val):
    if isinstance(val, _expr.StringImm):
        return val
    return _ir_adt.Constructor("String", ret_type=_type.StringType())(span, val)


def _to_unicode_ir(span, val):
    if isinstance(val, _expr.UnicodeImm):
        return val
    return _ir_adt.Constructor("Unicode", ret_type=_type.UnicodeType())(span, val)


# Builtin cast
_str2ir.checked_type = _type.StringType()
_str2ir.is_constructor = True
_to_unicode_ir.checked_type = _type.UnicodeType()
_to_unicode_ir.is_constructor = True
_register_op("str", _to_unicode_ir)
_register_op("bytes", _str2ir)

# Builtin Constructor
_register_container_cons("{}.runtime._container._dict.Dict".format(_module_name_), _type.DictType())
_register_container_cons("{}.runtime._container._list.List".format(_module_name_), _type.ListType())
_register_container_cons("{}.runtime._container._set.Set".format(_module_name_), _type.SetType())
_register_container_cons("{}.runtime.trie.Trie".format(_module_name_), _type.TrieType())
_register_container_cons(
    "{}.runtime._container._opaque_object.OpaqueObject".format(_module_name_),
    _type.OpaqueObjectType())
_register_pod_cons("bool", "bool", _type.PrimType("bool"))
_register_pod_cons("int", "int64_t", _type.PrimType("int64"))
_register_pod_cons("float", "double", _type.PrimType("float64"))
_register_builtin_container_cons("list", "List", _type.ListType())
_register_builtin_container_cons("set", "Set", _type.SetType())
_register_builtin_container_cons("dict", "Dict", _type.DictType())

# matx.xx_func
_register_op(make_native_object, _ir_op.matx_make_native_object)
_register_op("{}.native.make_native_op".format(_module_name_), _ir_op.matx_make_native_op)
_register_op(make_native_function, _ir_op.matx_make_native_function)
_register_op(call_native_function, _ir_op.matx_call_native_function)
_register_op("{}.pmap".format(_module_name_), _ir_op.matx_pmap)
_register_op("{}.pstarmap".format(_module_name_), _ir_op.matx_pstarmap)
_register_op("{}.apply_async".format(_module_name_), _ir_op.matx_apply_async)
_register_python_builtin("{}.runtime.picke.serialize".format(_module_name_), "pickle_serialize")
_register_python_builtin("{}.runtime.picke.deserialize".format(_module_name_), "pickle_deserialize")


def register_pypi_extension():
    from . import ops_farmhash
    Builtin2Op.registrations.update(ops_farmhash.registry_info)


register_pypi_extension()


def register_tuple_construct():
    def tuple_star_construct(span, *args):
        argtypes = [arg.checked_type for arg in args]
        retty = _type.TupleType(argtypes)
        cons = _ir_adt.Constructor("Tuple", inputs=argtypes, ret_type=retty)
        return cons(span, InitializerList(args))

    def tuple_construct(args):
        # builtin.tuple should be initialized with an iterable object, e.g.
        # tuple([1, 2, 3]), but we supports it in a different way that `fields`
        # should be unpacked *args, e.g. matx.Tuple(1, 2, 3) or (1, 2, 3).
        raise NotImplementedError(
            '`tuple(iterable)` is not supported now, please use it in literal, e.g. (1, 2, \'a\', ...).')

    tuple_construct.checked_type = _type.TupleType([])
    tuple_construct.is_constructor = True
    Builtin2Op.registrations["matx.runtime.container.Tuple"] = tuple_star_construct
    Builtin2Op.registrations["tuple_star"] = tuple_star_construct
    Builtin2Op.registrations["tuple"] = tuple_construct


register_tuple_construct()


def _register_regex_construct():
    def regex_construct(span,
                        pattern,
                        ignore_case=False,
                        dotall=False,
                        extended=False,
                        anchored=False,
                        ucp=True):
        return _ir_adt.Constructor(
            "Regex",
            ret_type=_type.RegexType())(
            span,
            pattern,
            ignore_case,
            dotall,
            extended,
            anchored,
            ucp)

    regex_construct.checked_type = _type.RegexType()
    regex_construct.is_constructor = True
    Builtin2Op.registrations["matx.runtime.regex.Regex"] = regex_construct


_register_regex_construct()


def _register_nd_array_construct():
    def nd_array_construct(span,
                           arr,
                           shape,
                           dtype,
                           device="cpu"):
        def convert_unicode(span, e, name):
            if isinstance(e.checked_type, _type.UnicodeType):
                return e
            elif isinstance(e.checked_type, _type.ObjectType):
                return _to_unicode_ir(span, e)
            else:
                raise TypeError(
                    "ndarray constructor: type of parameter %s should be Unicode, but get %s" %
                    (name, e.checked_type))

        if not isinstance(device, _expr.BaseExpr):
            assert isinstance(device, str), "internal error"
            device = _expr.UnicodeImm(device, span=span)
        # ndim = -1
        # if isinstance(shape, _expr.Call) and isinstance(shape.op, _ir_adt.Constructor):
        #     if len(shape.args) == 1:
        #         init = shape.args[0]
        #         if isinstance(init, _expr.InitializerList):
        #             ndim = len(init.fields)
        # checked_dtype = None
        # if isinstance(dtype, (_expr.StringImm, _expr.UnicodeImm)):
        #     checked_dtype = _type.PrimType(dtype.value)
        # ret_ty = _type.NDArrayType(ndim=ndim, dtype=checked_dtype)
        ret_ty = _type.NDArrayType()
        if isinstance(shape.checked_type, _type.ListType):
            pass
        elif isinstance(shape.checked_type, _type.ObjectType):
            shape = Builtin2Op.lookup("list")(span, shape)
        else:
            raise TypeError(
                "ndarray constructor: type of parameter shape should be list, but get %s" %
                shape.checked_type)

        dtype = convert_unicode(span, dtype, 'dtype')
        device = convert_unicode(span, device, 'device')

        return _ir_adt.Constructor("NDArray", ret_type=ret_ty)(
            span,
            arr,
            shape,
            dtype,
            device
        )

    nd_array_construct.checked_type = _type.NDArrayType()
    nd_array_construct.is_constructor = True
    Builtin2Op.registrations["matx.runtime.ndarray.NDArray"] = nd_array_construct


_register_nd_array_construct()
