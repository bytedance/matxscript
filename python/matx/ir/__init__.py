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
# pylint: disable=unused-import, redefined-builtin
"""Namespace for Tensor-level IR"""
from .constexpr import const, generic_const

from .base import SourceName, Span, Node
from .base import structural_equal, assert_structural_equal, structural_hash
from .base import BaseExpr, PrimExpr, HLOExpr
from .type import Type, TypeKind, PrimType, PointerType, TypeVar, GlobalTypeVar, TupleType
from .type import RangeType
from .type import FuncType
from .type import ObjectType, StringType, UnicodeType, ListType, DictType, SetType
from .type import IteratorType, FileType, VoidType, UserDataType
from .type import OpaqueObjectType
from .type import DynTensorType
from .type_relation import type_inference
from .op_expr import Op
from .function import CallingConv, BaseFunc

from .expr import PrimVar, FloatImm, IntImm, StringImm, PrimCast, HLOCastPrim, UnicodeImm
from .expr import PrimAdd, PrimSub, PrimMul, PrimDiv, PrimFloorDiv, PrimMod, PrimFloorMod
from .expr import PrimMin, PrimMax, PrimEQ, PrimNE, PrimLT
from .expr import PrimLE, PrimGT, PrimGE, PrimAnd, PrimOr, PrimNot
from .expr import PrimSelect, PrimCall, CallEffectKind, PrimLet
from .expr import GlobalVar, Call, InitializerList, InitializerDict
from .expr import HLOVar, NoneExpr, EnumAttr, HLOCast
from .expr import RangeExpr, TupleExpr
from .expr import Comprehension, ListComp, SetComp, DictComp

from .stmt import Stmt, AssertStmt, For, While, Break, Continue
from .stmt import SeqStmt
from .stmt import IfThenElse, stmt_seq, stmt_list
from .stmt import ReturnStmt, AssignStmt, AllocaVarStmt
from .stmt import ExprStmt, HLOYield, AutoFor
from .stmt import ExceptionHandler, TryExcept, Raise

from .tensor_stmt import Buffer, decl_buffer
from .tensor_stmt import BufferStore, BufferLoad, BufferRegion, MatchBufferRegion
from .tensor_stmt import ComputeBlock, ComputeBlockRealize, Allocate

from .function import PrimFunc, Function, LambdaFunction, FuncAttr
from .class_stmt import ClassStmt
from .module import IRModule

from .op import call_intrin, call_extern
from .op import builtins_all, builtins_any, builtins_abs, builtins_round
from .op import min_value, max_value
from .op import math_exp, matx_math_exp2, matx_math_exp10
from .op import math_log, math_log2, math_log10, math_log1p, math_ldexp
from .op import math_sin, math_sinh, math_asin, math_asinh
from .op import math_cos, math_cosh, math_acos, math_acosh
from .op import math_tan, math_tanh, math_atan, math_atan2, math_atanh
from .op import math_erf, math_sqrt, math_floor, math_ceil, math_hypot
from .op import matx_math_sigmoid, matx_math_rsqrt, matx_math_nearbyint
from .op import math_trunc, math_nextafter, math_pow, math_fmod
from .op import math_isnan, math_isfinite, math_isinf, math_copysign
from .op import if_then_else, popcount
from .op import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from .op import q_multiply_shift
from .op import builtins_unpack

from .adt import Constructor, ClassType

from . import ir_builder
from . import analysis

from .builtin2op import Builtin2Op
