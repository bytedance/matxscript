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
from .base import BaseExpr, PrimExpr, HLOExpr, Range
from .type import Type, TypeKind, PrimType, PointerType, TypeVar, GlobalTypeVar, TupleType
from .type import FuncType
from .type import ObjectType, StringType, UnicodeType, ListType, DictType, SetType
from .type import IteratorType, FileType, VoidType, UserDataType
from .type import OpaqueObjectType
from .type import NDArrayType
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

from .stmt import Stmt, LetStmt, AssertStmt, For, While, Break, Continue
# from .stmt import BufferStore, BufferRealize, Store, ProducerStore, Allocate, AttrStmt
from .stmt import SeqStmt
from .stmt import IfThenElse, Evaluate, stmt_seq, stmt_list
from .stmt import ReturnStmt, AssignStmt, AllocaVarStmt
from .stmt import ExprStmt, HLOYield, AutoFor
from .stmt import ExceptionHandler, TryExcept, Raise

from .function import PrimFunc, Function, LambdaFunction, FuncAttr
from .module import IRModule

from .op import call_intrin, call_extern
from .op import all, any, min_value, max_value
from .op import exp, exp2, exp10, log, log2, log10, log1p, ldexp
from .op import sin, sinh, asin, asinh
from .op import cos, cosh, acos, acosh
from .op import tan, tanh, atan, atan2, atanh
from .op import erf, sigmoid, sqrt, rsqrt, floor, ceil, hypot
from .op import trunc, abs, round, nextafter, nearbyint, power, popcount, fmod, if_then_else
from .op import isnan, isfinite, isinf, copysign
from .op import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from .op import q_multiply_shift
from .op import builtins_unpack

from .adt import Constructor, ClassType

from . import ir_builder
from . import analysis

from .builtin2op import Builtin2Op
