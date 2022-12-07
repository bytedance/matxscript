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

import copy
import inspect
import random
import builtins
import sys
import numbers

from typing import Any, Union, List, Optional
from matx._typed_ast import ast

from .reporter.script_error import MATXScriptError

from . import context
from . import utils
from .. import ir as _ir
from ..ir import generic as _generic
from ..ir import type_inference as _type_infer
from ..ir import type_relation as _type_rel
from ..ir import Builtin2Op
from .type_parser import parse_type
from .context import Span
from ..pipeline import PluginLoader
from ..pipeline.ops import OpKernel
from ..pipeline.jit_object import JitObject, JitOpImpl
from ..native import NativeFunction
from ..native import NativeClass
from .analysis import LiveVariableAnalysis

NAME_NOT_FOUND = object()
MATX_MODULE = sys.modules["matx"]


class CollectLocalVars(ast.NodeVisitor):
    def __init__(self, ctx: context.ScopeContext, init_free_vars=None) -> None:
        self._ctx = ctx
        self._vars = init_free_vars or []

    def _not_exists(self, sym):
        for s in self._vars:
            if s.same_as(sym):
                return False
        return True

    def run(self, node: ast.AST):
        self.visit(node)
        return self._vars

    def visit_Name(self, node: ast.Name):
        name = node.id
        symbol = self._ctx.lookup_symbol(name)
        if symbol is not None and self._not_exists(symbol):
            self._vars.append(symbol)


def get_free_vars(func, accept_types):
    free_vars = {}
    if inspect.ismethod(func):
        func = func.__func__

    if not inspect.isfunction(func):
        raise TypeError("{!r} is not a Python function".format(func))

    code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    if func.__closure__ is not None:
        for var, cell in zip(code.co_freevars, func.__closure__):
            try:
                if isinstance(cell.cell_contents, accept_types):
                    free_vars[var] = cell.cell_contents
            except:
                # TODO: check why raise exception ?
                pass

    # Global and builtin references are named in co_names and resolved
    # by looking them up in __globals__ or __builtins__
    global_ns = func.__globals__
    builtin_ns = global_ns.get("__builtins__", builtins.__dict__)
    if inspect.ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__
    for name in code.co_names:
        if name in ("None", "True", "False"):
            # Because these used to be builtins instead of keywords, they
            # may still show up as name references. We ignore them.
            continue
        try:
            if isinstance(global_ns[name], accept_types):
                free_vars[name] = global_ns[name]
        except KeyError:
            try:
                if isinstance(builtin_ns[name], accept_types):
                    free_vars[name] = builtin_ns[name]
            except KeyError:
                # ignore unbound_names
                pass

    return free_vars


class CallArgumentReader(object):
    """A helper class which read required argument from passed arguments"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_pos_only_arg(self, pos, name):
        """Get corresponding position only function argument from argument list"""
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs:
            self.parser.report_error('{} misses argument {}'.format(self.func_name, name),
                                     SyntaxError)
        else:
            arg = self.kwargs[name]

        return arg

    def get_kwarg(self, pos, name, default):
        """Get corresponding keyword function argument from argument list
        If user doesn't provide the argument, set it to default value
        """
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name in self.kwargs:
            arg = self.kwargs[name]
        else:
            return default

        return arg

    def get_varargs(self, pos):
        """Get corresponding variable argument from argument list"""
        if len(self.args) >= pos and len(self.kwargs) == 0:
            return self.args[pos - 1:]
        return []


def try_to_view_type(origin_type: _ir.Type):
    new_type = origin_type
    success = False
    if isinstance(origin_type, _ir.UnicodeType):
        new_type = _ir.UnicodeType(is_view=True)
        success = True
    if isinstance(origin_type, _ir.StringType):
        new_type = _ir.StringType(is_view=True)
        success = True
    if isinstance(origin_type, _ir.ObjectType):
        new_type = _ir.ObjectType(is_view=True)
        success = True
    return success, new_type


class MATXScriptParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to MATX's IR
    Notes for extension:
    1. To support new types of AST nodes. Add a function visit_xxx().
    """

    _binop_maker = {
        ast.Add: lambda lhs, rhs, span: _generic.add(lhs, rhs, span),
        ast.Sub: lambda lhs, rhs, span: _generic.subtract(lhs, rhs, span),
        ast.Mult: lambda lhs, rhs, span: _generic.multiply(lhs, rhs, span),
        ast.Div: lambda lhs, rhs, span: _generic.divide(lhs, rhs, span),
        ast.FloorDiv: lambda lhs, rhs, span: _generic.floordiv(lhs, rhs, span),
        ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
        ast.BitOr: lambda lhs, rhs, span: _generic.bitwise_or(lhs, rhs, span),
        ast.BitAnd: lambda lhs, rhs, span: _generic.bitwise_and(lhs, rhs, span),
        ast.BitXor: lambda lhs, rhs, span: _generic.bitwise_xor(lhs, rhs, span),
        ast.LShift: lambda lhs, rhs, span: _generic.left_shift(lhs, rhs, span),
        ast.RShift: lambda lhs, rhs, span: _generic.right_shift(lhs, rhs, span),
        ast.Gt: lambda lhs, rhs, span: _generic.greater_than(lhs, rhs, span),
        ast.GtE: lambda lhs, rhs, span: _generic.greater_or_equal(lhs, rhs, span),
        ast.Lt: lambda lhs, rhs, span: _generic.less_than(lhs, rhs, span),
        ast.LtE: lambda lhs, rhs, span: _generic.less_or_equal(lhs, rhs, span),
        ast.Eq: lambda lhs, rhs, span: _generic.equal(lhs, rhs, span),
        ast.NotEq: lambda lhs, rhs, span: _generic.notequal(lhs, rhs, span),
        ast.Is: lambda lhs, rhs, span: _generic.op_is(lhs, rhs, span),
        ast.IsNot: lambda lhs, rhs, span: _generic.op_not(_generic.op_is(lhs, rhs, span), span)
    }

    _unaryop_maker = {
        ast.USub: lambda operand, span: _generic.multiply(operand, _ir.const(-1), span),
        ast.Invert: lambda operand, span: _generic.bitwise_not(operand, span),
        ast.Not: lambda operand, span: _generic.op_not(operand, span)
    }

    _boolop_marker = {
        ast.And: lambda span, *args: _generic.op_and(span, *args),
        ast.Or: lambda span, *args: _generic.op_or(span, *args)
    }

    def __init__(self, node: context.ASTNode, sc_ctx: context.ScriptContext):
        self.fn_is_generator = False
        self.custom_ast_node = node
        self.sc_ctx = sc_ctx
        if isinstance(node.context, context.FunctionContext):
            self.fn_ctx = node.context
        else:
            self.fn_ctx = None
        self.context = None
        self.fn_live_out_variables = dict()
        self.functions = {}
        self.current_node = None
        self.parent_node = None
        self.class_mode = False
        self._visit_call_func = False
        self._random = random.Random(201208)
        self.class_or_func_type = node.ir_schema
        self.class_instance_var = None
        self.session_handle_var_ctx = []
        self.ann_type_context = [None]

    def push_ann_type(self, ty):
        self.ann_type_context.append(ty)

    def pop_ann_type(self):
        self.ann_type_context.pop()

    @property
    def current_ann_type(self) -> Optional[_ir.Type]:
        return self.ann_type_context[-1]

    def push_handle_var(self, handle_var):
        self.session_handle_var_ctx.append(handle_var)

    def pop_handle_var(self):
        self.session_handle_var_ctx.pop()

    @property
    def current_handle_var(self) -> Optional[_ir.PrimVar]:
        return self.session_handle_var_ctx[-1]

    def gen_random(self):
        # The first 8178 numbers are unique with random seed of 201208, I think it's enough.
        return self._random.randrange(10e6, 10e7)

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.context = context.ScopeContext()

    def build_span(self, node, with_error=False):
        # TODO: build self.custom_ast_node.span in source analysis
        root_span = self.custom_ast_node.span
        abs_lineno = root_span.lineno + node.lineno - 1
        source_code = root_span.source_code

        if with_error:
            new_reporter_span = Span()
            new_reporter_span.file_name = root_span.file_name
            new_reporter_span.lineno = abs_lineno
            new_reporter_span.func_name = self.custom_ast_node.context.name
            new_reporter_span.source_code = source_code
            return new_reporter_span

        return _ir.Span(root_span.file_name,
                        abs_lineno,
                        self.custom_ast_node.context.name,
                        source_code)

    def visit(self, node: ast.AST) -> Any:
        """Override method in ast.NodeVisitor"""
        last_parent_node = self.parent_node
        last_cur_node = self.current_node
        self.parent_node = self.current_node
        self.current_node = node
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        try:
            visit_res = visitor(node)
        except MATXScriptError:
            raise
        except Exception as e:
            self.report_error(str(e), type(e))
        self.current_node = last_cur_node
        self.parent_node = last_parent_node

        return visit_res

    def report_error(self, err_msg: str, err_type: Exception = None):
        self.sc_ctx.reporter.error(
            err_msg,
            self.build_span(self.current_node, True),
            err_type
        )

    def parse_body(self, auto_add_return=False):
        body = []
        last_ast = None
        while len(self.context.node_stack[-1]) > 0:
            last_ast = self.context.node_stack[-1].pop()
            res = self.visit(last_ast)
            if res is not None:
                if not isinstance(res, _ir.Stmt):
                    self.report_error('Every IR node here should be a stmt!')
                body.append(res)
            else:
                # ignore the stmt
                pass
        if (auto_add_return
                and (not self.fn_is_generator)
                and (len(body) == 0 or not isinstance(last_ast, ast.Return))):
            body.append(self.visit(ast.Return(value=None)))
        return body

    @staticmethod
    def to_seq_stmt(body: List[_ir.Stmt], span: _ir.Span):
        if body is None or len(body) == 0:
            return _ir.SeqStmt(body, span)
        return _ir.SeqStmt(body, span) if len(body) > 1 else body[0]

    @staticmethod
    def parse_docstring(node: Union[ast.FunctionDef, ast.ClassDef]):
        docstring = None
        if (len(node.body) > 0
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)):
            docstring = node.body[0].value.s
        return docstring

    def parse_arg_list(self, func, node_call):
        assert isinstance(node_call, ast.Call)
        # collect arguments
        args = [self.visit(arg) for arg in node_call.args]
        kw_args = [self.visit(keyword) for keyword in node_call.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        # get the name and parameter list of func
        func_name = func.__name__
        param_list = utils.get_param_list(func)
        # check arguments and parameter list and get a list of arguments
        reader = CallArgumentReader(func_name, args, kw_args, self)
        pos_only, kwargs, varargs = param_list
        internal_args = list()
        for i, arg_name in enumerate(pos_only):
            internal_args.append(reader.get_pos_only_arg(i + 1, arg_name))
        for i, arg_info in enumerate(kwargs):
            arg_name, default = arg_info
            internal_args.append(reader.get_kwarg(i + 1 + len(pos_only), arg_name, default=default))
        if varargs is not None:
            internal_args.extend(reader.get_varargs(len(pos_only) + len(kwargs) + 1))
        return internal_args

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """
        self.report_error('This node is not supported now: {}'.format(node), NotImplementedError)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        """FormattedValue visitor
        """
        func_name = "str"
        to_str_func = Builtin2Op.lookup(func_name)
        span = self.build_span(node)
        visited_result = self.visit(node.value)
        if isinstance(visited_result.checked_type, _ir.ClassType):
            symbol = visited_result
            dep_node_ctx = self.sc_ctx.get_ast_node_by_class_type(symbol.checked_type)
            if "__str__" in dep_node_ctx.context.methods:
                func = dep_node_ctx.context.ir_call_attr(span, symbol, "__str__")
                visited_result = func(span)
            elif "__repr__" in dep_node_ctx.context.methods:
                func = dep_node_ctx.context.ir_call_attr(span, symbol, "__repr__")
                visited_result = func(span)
        pos_args = [visited_result]
        kw_args = {}
        return to_str_func(span, *pos_args, **kw_args)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """JoinedStr visitor
        By now only basic pattern is supported.
        No advance syntax or notation such as the alignment syntax and the notation for float number is supported
        """
        visited_node = [self.visit(n) for n in node.values]
        if len(visited_node) == 0:
            self.report_error('empty joined str')
        elif len(visited_node) == 1:
            return visited_node[0]
        curr = visited_node[0]
        span = self.build_span(node)
        string_add = MATXScriptParser._binop_maker[ast.Add]
        for i in range(1, len(visited_node)):
            curr = string_add(curr, visited_node[i], span)
        return curr

    def visit_Module(self, node):
        """Module visitor"""

        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        self.report_error('Only one-function, one-class source code is allowed')

    def visit_ClassDef(self, node: ast.ClassDef):
        """ClassDef visitor
        AST abstract grammar:
            ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body,
                     expr* decorator_list)
        """
        # https://docs.python.org/3/library/ast.html#ast.ClassDef
        class_mode = self.class_mode
        self.class_mode = True
        cls_ctx = self.custom_ast_node.context
        last_class_ins_var = self.class_instance_var
        self.class_instance_var = cls_ctx.class_instance_var
        self.push_handle_var(cls_ctx.session_pointer_var)
        # parse docstring
        docstring = self.parse_docstring(node)
        # parse member functions
        for elem in node.body:
            if isinstance(elem, ast.FunctionDef):
                func_name = elem.name
                if func_name == "__init__":
                    self.fn_ctx = cls_ctx.init_fn
                else:
                    self.fn_ctx = cls_ctx.methods[elem.name]
                unbound_name = self.fn_ctx.unbound_name
                bound_name = self.fn_ctx.name
                elem.name = unbound_name
                ir_func = self.visit(elem)
                ir_func = ir_func.with_attr({_ir.FuncAttr.kClassNameBelongTo: node.name})
                ir_func = ir_func.with_attr({_ir.FuncAttr.kBoundSymbol: bound_name})
                if func_name == "__init__":
                    ir_func = ir_func.with_attr({_ir.FuncAttr.kClassConstructor: True})
                self.functions[unbound_name] = ir_func
        # parse other
        # gen result
        ir_module = _ir.IRModule(self.functions, {node.name: self.class_or_func_type})
        self.class_mode = class_mode
        self.class_instance_var = last_class_ins_var
        self.pop_handle_var()
        return ir_module

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """FunctionDef visitor
        AST abstract grammar:
            FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list,
                        expr? returns, string? type_comment)
            arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                         expr* kw_defaults, arg? kwarg, expr* defaults)
            arg = (identifier arg, expr? annotation, string? type_comment)
        """
        lva = LiveVariableAnalysis(node)
        self.fn_live_out_variables = lva.get_live_out_mapping()
        self.fn_is_generator = False
        docstring = self.parse_docstring(node)
        self.init_function_parsing_env()
        self.context.new_scope(nodes=node.body)
        span = self.build_span(node)
        # add parameters of function
        for arg_pos, arg in enumerate(node.args.args):
            if (arg_pos == 0
                    and self.class_mode
                    and self.fn_ctx.fn_type == context.FunctionType.INSTANCE):  # TODO: check this
                arg_var = self.class_instance_var
            else:
                # var_type = parse_type(arg.annotation, self.custom_ast_node, self.sc_ctx)
                var_type = self.fn_ctx.arg_types[arg.arg]
                if isinstance(var_type, _ir.PrimType):
                    arg_var = _ir.PrimVar(arg.arg, var_type, span)
                else:
                    arg_var = _ir.HLOVar(arg.arg, var_type, span)
            self.context.update_symbol(arg.arg, arg_var)
            self.context.func_params.append(arg_var)

        is_class_init = self.class_mode and self.fn_ctx.name == "__init__"
        is_function = not self.class_mode
        if is_class_init or is_function:
            # append session_pointer_var
            pointer_var_name = "handle_2_71828182846"
            pointer_var = _ir.PrimVar(
                pointer_var_name,
                _ir.PrimType("handle")
            )
            self.context.update_symbol(pointer_var_name, pointer_var)
            self.context.func_params.append(pointer_var)
        if is_function:
            self.push_handle_var(self.context.func_params[-1])
        argdefaults = [self.visit(x) for x in node.args.defaults]
        ret_type = parse_type(node.returns, self.custom_ast_node, self.sc_ctx)

        self.context.func_ret_type = ret_type

        for i in range(len(argdefaults)):
            if not isinstance(argdefaults[i], _ir.BaseExpr):
                argdefaults[i] = _ir.generic_const(argdefaults[i])
        alloca_closure_var_stmts = []
        free_vars = get_free_vars(self.fn_ctx.raw, (OpKernel,))
        for cap_name, cap_var in free_vars.items():
            if isinstance(cap_var, OpKernel):
                # add to script context
                self.sc_ctx.add_free_var(cap_var)
                # alloca capture var
                closure_var_ty = _ir.ObjectType(is_view=True)
                init_value = _ir.call_extern(closure_var_ty,
                                             "GetClosureVar",
                                             span,
                                             self.current_handle_var,
                                             _ir.StringImm(cap_var.native_class_name),
                                             _ir.StringImm(cap_var.name))
                ass = _ir.AllocaVarStmt(cap_name, closure_var_ty, init_value, span)
                alloca_closure_var_stmts.append(ass)
                self.context.update_symbol(cap_name, ass.var)
        body_stmts = self.parse_body(True)
        body_stmts = alloca_closure_var_stmts + body_stmts
        if is_class_init or is_function:
            # append defaults
            argdefaults.append(_ir.PrimCast("handle", _ir.const(0)))
        if is_class_init:
            # assign session pointer var
            ass = _ir.AssignStmt(self.current_handle_var, self.context.func_params[-1], span)
            body_stmts.insert(0, ass)

        # fetch the body and return a tir.PrimFunc
        func = _ir.Function(
            self.context.func_params,
            argdefaults,
            self.to_seq_stmt(body_stmts, span),
            ret_type=ret_type,
            span=span
        )
        func = func.with_attr(_ir.FuncAttr.kGlobalSymbol, node.name)
        if is_class_init or is_function:
            func = func.with_attr({_ir.FuncAttr.kCaptureSessionHandle: True})
        self.functions[node.name] = func

        if is_function:
            self.pop_handle_var()
        self.context.pop_scope()
        return func

    def visit_Return(self, node):
        if node.value is None:
            rt_expr = _type_rel.smart_adapt_to(_ir.NoneExpr(), self.context.func_ret_type)
            return _ir.ReturnStmt(rt_expr)
        span = self.build_span(node)
        self.push_ann_type(self.context.func_ret_type)
        rt_expr = self.visit(node.value)
        self.pop_ann_type()
        if isinstance(rt_expr, tuple):
            rt_expr = Builtin2Op.lookup("matx.runtime.container.Tuple")(*rt_expr)
        if not _type_rel.type_convertible(
                rt_expr.checked_type,
                self.context.func_ret_type):
            self.report_error(
                'Cannot return `{}` in function with annotation of `{}` as return type.'.format(
                    rt_expr.checked_type.py_type_name(),
                    self.context.func_ret_type.py_type_name()),
                SyntaxError)
        rt_expr = _type_rel.smart_adapt_to(rt_expr, self.context.func_ret_type, span)
        return _ir.ReturnStmt(rt_expr, span)

    def lookup_or_alloca(self, name_hint, init_value, node, ann_ty=None, live_out=True):
        span = self.build_span(node)
        init_value = user_function_wrapper(init_value, self.current_handle_var, span)
        inf_ty = _type_infer(init_value)
        if (isinstance(inf_ty, (_ir.StringType, _ir.UnicodeType, _ir.ObjectType,))
                and inf_ty.is_view):
            inf_ty = type(inf_ty)(is_view=False)
        symbol, level = self.context.lookup_symbol_with_level(name_hint)
        if ann_ty is not None:
            ann_ty_name = ann_ty.py_type_name()
            if symbol is not None and symbol.py_type_name() != ann_ty_name:
                self.report_error(
                    "redeclare '{}' type from '{}' to '{}' is not supported".format(
                        name_hint,
                        symbol.py_type_name(),
                        ann_ty.get_py_type_name()),
                    TypeError
                )
            if not _type_rel.type_convertible(inf_ty, ann_ty):
                self.report_error(
                    "name '{}' is declared as '{}' type, but right value is '{}' type".format(
                        name_hint,
                        ann_ty.get_py_type_name(),
                        inf_ty.get_py_type_name()),
                    TypeError
                )
            inf_ty = ann_ty
            init_value = _type_rel.smart_adapt_to(init_value, inf_ty, span)
        if symbol is None:
            alloca_stmt = _ir.AllocaVarStmt(name_hint, inf_ty, init_value, span)
            self.context.update_symbol(name_hint, alloca_stmt.var)
            return alloca_stmt
        else:
            if not _type_rel.type_convertible(inf_ty, symbol.checked_type):
                if level == -1 or (not live_out):
                    # deref last var and alloca a new var
                    new_name_hint = name_hint + "_ssa_" + str(self.gen_random())
                    alloca_stmt = _ir.AllocaVarStmt(new_name_hint, inf_ty, init_value, span)
                    self.context.update_symbol(name_hint, alloca_stmt.var)
                    return alloca_stmt
                self.report_error(
                    "name '{}' is declared as '{}' type, but right value is '{}' type".format(
                        name_hint,
                        symbol.py_type_name(),
                        inf_ty.get_py_type_name()),
                    TypeError
                )
            holder = self.context.lookup_referenced_symbol(symbol)
            if holder is None:
                init_value = _type_rel.smart_adapt_to(init_value, symbol.checked_type, span)
                return _ir.AssignStmt(symbol, init_value, span)
            else:
                init_value = _type_rel.smart_adapt_to(init_value, holder.checked_type, span)
                holder_assign = _ir.AssignStmt(holder, init_value, span)
                symbol_assign = _ir.AssignStmt(symbol, holder, span)
                return _ir.SeqStmt([holder_assign, symbol_assign], span)

    def visit_Assign(self, node: ast.Assign):
        """Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)

        By now 4 patterns of Assign is supported:
            1. special stmts with return value
                1.1 tmp = matx.List() # matx.Dict() matx.Set()...
                1.2 tmp = 3 # scalar type
                1.3 tmp = var
            2. (container ops) List[Expr] = Expr
                2.1 tmp["hi"] = 30 # tmp is matx.Dict

            3. (container ops) tmp = List[Expr]
            4. (container ops) tmp[xx] = List[xx]
            5. unpack assign
                5.1 a, b, c = (1, 2, 3)
        """
        # https://docs.python.org/3/library/ast.html#ast.Assign
        if not len(node.targets) == 1:
            # a = b = c
            self.report_error('Only one-valued assignment is supported now', NotImplementedError)

        span = self.build_span(node)

        def lhs_name_assign(lhs_node, rhs_val_eval_func):
            lhs_name = lhs_node.id
            # Pattern 1 & 3
            symbol = self.context.lookup_symbol(lhs_name)
            if symbol is None:
                rhs_val = rhs_val_eval_func()
            else:
                rhs_val = rhs_val_eval_func(symbol.checked_type)
            live_out = self.fn_live_out_variables.get(lhs_node, True)
            return self.lookup_or_alloca(lhs_name, rhs_val, lhs_node, live_out=live_out)

        def lhs_subscript_attr_assign(lhs_node, rhs_val_eval_func):
            # Pattern 2 & 4
            import types
            lhs_value = self.visit(lhs_node)
            if isinstance(lhs_value, types.FunctionType):
                exp_ty = None
                if hasattr(lhs_value, 'checked_type'):
                    if isinstance(lhs_value.checked_type, _ir.type.ListType):
                        exp_ty = lhs_value.checked_type.item_type
                    elif isinstance(lhs_value.checked_type, _ir.type.DictType):
                        exp_ty = lhs_value.checked_type.value_type
                if not (exp_ty and exp_ty.is_full_typed()):
                    exp_ty = None
                rhs_val = rhs_val_eval_func(exp_ty)
                return _ir.ExprStmt(lhs_value(rhs_val), span)
            else:
                rhs_val = rhs_val_eval_func(lhs_value.checked_type)
                if not _type_rel.type_convertible(rhs_val.checked_type, lhs_value.checked_type):
                    self.report_error(
                        "invalid conversion from '{}' to '{}'".format(
                            rhs_val.py_type_name(),
                            lhs_value.py_type_name()
                        ),
                        TypeError
                    )
                rhs_val = _type_rel.smart_adapt_to(rhs_val, lhs_value.checked_type, span)
                return _ir.AssignStmt(lhs_value, rhs_val, span)

        def lhs_tuple_unroll_assign(lhs_node: ast.Tuple, rhs_node: ast.Tuple):
            def build_seqs(lhs_node: ast.Tuple, rhs_node: ast.Tuple):
                if len(lhs_node.elts) != len(rhs_node.elts):
                    self.report_error('not enough values to unpack', ValueError)

                seq1 = []
                seq2 = []
                for i, (lhs_elt, rhs_elt) in enumerate(zip(lhs_node.elts, rhs_node.elts)):
                    rhs_name = 'unroll_{}_{}'.format(i, self.gen_random())
                    if isinstance(lhs_elt, ast.Tuple) and isinstance(rhs_elt, ast.Tuple):
                        seq1_, seq2_ = build_seqs(lhs_elt, rhs_elt)
                        seq1.extend(seq1_)
                        seq2.extend(seq2_)
                    else:
                        rhs_name_assign = self.lookup_or_alloca(
                            rhs_name, self.visit(rhs_elt), lhs_node)
                        lhs_assign = assign_dispatch_value(
                            lhs_elt, lambda *args: rhs_name_assign.var
                        )
                        seq1.append(rhs_name_assign)
                        seq2.append(lhs_assign)
                return seq1, seq2

            seq1, seq2 = build_seqs(lhs_node, rhs_node)
            return _ir.SeqStmt(seq1 + seq2, span)

        def lhs_tuple_unpack_assign(lhs_node, rhs_val):
            # pattern 5
            seq_stmts = []
            need_cache = isinstance(rhs_val, _ir.Call)
            lhs_node_elts = lhs_node.elts
            lhs_value_num = len(lhs_node_elts)
            if need_cache:
                cached_rhs_name = 'unpack_' + str(self.gen_random())
                cache_assign = self.lookup_or_alloca(cached_rhs_name, rhs_val, lhs_node)
                rhs_val = cache_assign.var
            rhs_values = []
            for i in range(lhs_value_num):
                rhs_values.append(_ir.op.object_get_item(span, rhs_val, _ir.const(i, "int64")))
            rhs_value_len = _ir.op.object_len(span, rhs_val)
            cond = rhs_value_len == _ir.const(lhs_value_num, "int64")
            msg = "ValueError: values to unpack mismatch (expected %d)" % lhs_value_num
            for i in range(lhs_value_num):
                rhs_eval_func = lambda *args: rhs_values[i]
                seq_stmts.append(assign_dispatch_value(lhs_node_elts[i], rhs_eval_func))
            if need_cache:
                return _ir.SeqStmt([cache_assign, _ir.AssertStmt(
                    cond, msg, _ir.SeqStmt(seq_stmts))], span)
            else:
                return _ir.AssertStmt(cond, msg, _ir.SeqStmt(seq_stmts), span)

        def assign_dispatch_value(lhs_node, rhs_val_eval_func):
            if isinstance(lhs_node, ast.Name):
                return lhs_name_assign(lhs_node, rhs_val_eval_func)
            elif isinstance(lhs_node, (ast.Subscript, ast.Attribute)):
                return lhs_subscript_attr_assign(lhs_node, rhs_val_eval_func)
            elif isinstance(lhs_node, ast.Tuple):
                return lhs_tuple_unpack_assign(lhs_node, rhs_val_eval_func())
            else:
                self.report_error('Unsupported Assign statement', NotImplementedError)

        def assign_dispatch_node(lhs_node, rhs_node):
            if isinstance(lhs_node, ast.Tuple) and isinstance(rhs_node, ast.Tuple):
                return lhs_tuple_unroll_assign(lhs_node, rhs_node)
            else:
                def eval_rhs_value(exp_ty=None):
                    self.push_ann_type(exp_ty)
                    rhs_value = self.visit(rhs_node)
                    if isinstance(rhs_node, ast.Name):
                        rhs_value = user_function_wrapper(rhs_value, self.current_handle_var, span)
                    self.pop_ann_type()
                    return rhs_value

                return assign_dispatch_value(lhs_node, eval_rhs_value)

        return assign_dispatch_node(node.targets[0], node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """AnnAssign visitor
        We will enable type annotations only when we need to define variables, otherwise
        fallback to the implementation of ast.Assign
        """
        span = self.build_span(node)
        if isinstance(node.target, ast.Name):
            lhs_node = node.target
            lhs_name = lhs_node.id
            ann_type = parse_type(node.annotation, self.custom_ast_node, self.sc_ctx)
            self.push_ann_type(ann_type)
            rhs_val = self.visit(node.value)
            self.pop_ann_type()
            if isinstance(node.value, ast.Name):
                # TODO: clean code
                rhs_val = user_function_wrapper(rhs_val, self.current_handle_var, span)
            live_out = self.fn_live_out_variables.get(lhs_node, True)
            return self.lookup_or_alloca(lhs_name, rhs_val, lhs_node, ann_type, live_out=live_out)
        else:
            assign_ast = ast.Assign(
                targets=[
                    node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset)
            return self.visit_Assign(assign_ast)

    def visit_AugAssign(self, node: ast.AugAssign):
        info = {'lineno': node.lineno, 'col_offset': node.col_offset}
        lhs = copy.deepcopy(node.target)
        lhs.ctx = ast.Load()
        binop = ast.BinOp(lhs, node.op, node.value, **info)
        return self.visit(ast.Assign([node.target], binop, None, **info))

    def visit_Assert(self, node):
        """Assert visitor
        AST abstract grammar:
            Assert(expr test, expr? msg)

        Pattern corresponds to concise mode of with tir.Assert()
        """

        span = self.build_span(node)
        condition = self.visit(node.test)
        if node.msg is None:
            self.report_error('Message of AssertStmt can\'t be None', SyntaxError)
        message = self.visit(node.msg)
        body = self.to_seq_stmt(self.parse_body(), span)
        return _ir.AssertStmt(condition, node.msg.s, body, span)

    def check_loop_vars(self, targets: ast.Tuple, iter_expr: _ir.BaseExpr):
        if isinstance(targets, ast.Tuple):
            num_ele = len(targets.elts)
            if isinstance(iter_expr, _ir.expr.HLOEnumerate):
                if num_ele > 2:
                    self.report_error(
                        f'not enough values to unpack (expected 2, got {num_ele})',
                        ValueError
                    )
            elif isinstance(iter_expr, _ir.expr.HLOZip):
                num_it_args = len(iter_expr.values)
                if num_ele > num_it_args:
                    self.report_error(
                        f'not enough values to unpack (expected {num_it_args}, got {num_ele})',
                        ValueError
                    )
                elif num_ele < num_it_args:
                    self.report_error(
                        f'too many values to unpack (expected {num_it_args}, got {num_ele})',
                        ValueError
                    )

    def visit_For(self, node: ast.For):
        span = self.build_span(node)
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id in ("range", "xrange")):
            if isinstance(node.target, ast.Tuple):
                self.report_error(
                    'cannot unpack non-iterable int object',
                    TypeError,
                )
            if not isinstance(node.target, ast.Name):
                self.report_error(
                    f'cannot assign to operator',
                    SyntaxError,
                )
            symbol = self.context.lookup_symbol(node.target.id)
            if symbol is not None:
                self.report_error("name '%s' is redefined in for_range" % node.target.id)
            # LoopRange
            var = _ir.PrimVar(node.target.id, _ir.PrimType('int64'))
            if len(node.iter.args) == 1:
                start = _ir.const(0, 'int64')
                stop = self.visit(node.iter.args[0])
                step = _ir.const(1, 'int64')
            elif len(node.iter.args) == 2:
                start = self.visit(node.iter.args[0])
                stop = self.visit(node.iter.args[1])
                step = _ir.const(1, 'int64')
            elif len(node.iter.args) == 3:
                start = self.visit(node.iter.args[0])
                stop = self.visit(node.iter.args[1])
                step = self.visit(node.iter.args[2])
            else:  # error report
                if len(node.iter.args) == 0:
                    self.report_error("No arguments supplied in range expression.")
                else:
                    self.report_error(
                        "Invalid arguments in range expression")

            self.context.new_scope(nodes=node.body)
            self.context.update_symbol(node.target.id, var)
            forbody = self.to_seq_stmt(self.parse_body(), span)
            self.context.pop_scope()
            tmp_stmts = []
            if not isinstance(start, (_ir.IntImm, _ir.FloatImm, _ir.PrimVar)):
                tmp_name = "%s_start_%d" % (node.target.id, self.gen_random())
                start_stmt = _ir.AllocaVarStmt(tmp_name, _ir.PrimType('int64'), start, span)
                start = start_stmt.var
                tmp_stmts.append(start_stmt)
            if not isinstance(stop, (_ir.IntImm, _ir.FloatImm, _ir.PrimVar)):
                tmp_name = "%s_stop_%d" % (node.target.id, self.gen_random())
                stop_stmt = _ir.AllocaVarStmt(tmp_name, _ir.PrimType('int64'), stop, span)
                stop = stop_stmt.var
                tmp_stmts.append(stop_stmt)
            if not isinstance(step, (_ir.IntImm, _ir.FloatImm, _ir.PrimVar)):
                tmp_name = "%s_step_%d" % (node.target.id, self.gen_random())
                step_stmt = _ir.AllocaVarStmt(tmp_name, _ir.PrimType('int64'), step, span)
                step = step_stmt.var
                tmp_stmts.append(step_stmt)
            stmt = _ir.For(
                var,
                _generic._cast_to_prim_int(start, span),
                _generic._cast_to_prim_int(stop, span),
                _generic._cast_to_prim_int(step, span),
                0,
                1,
                forbody,
                span)
            if tmp_stmts:
                tmp_stmts.append(stmt)
                return _ir.SeqStmt(tmp_stmts, span)
            return stmt
        else:
            # AutoFor
            iter_expr = self.visit(node.iter)
            iter_expr_type = iter_expr.checked_type
            self.context.new_scope(nodes=node.body)
            if isinstance(node.target, ast.Name):
                symbol = self.context.lookup_symbol(node.target.id)
                if symbol is not None:
                    self.report_error("name '%s' is redefined in AutoFor" % node.target.id)
                if isinstance(iter_expr_type, _ir.NDArrayType):
                    # eval iter expr
                    alloca_iter_expr_var = None
                    if not isinstance(node.iter, ast.Name):
                        iter_expr_var_name = '__reserved_tmp_iter_expr_{}'.format(
                            self.gen_random()
                        )
                        alloca_iter_expr_var = _ir.AllocaVarStmt(
                            iter_expr_var_name, _ir.NDArrayType(), iter_expr, span
                        )
                        iter_expr = alloca_iter_expr_var.var
                    iter_var_hash = abs(hash(node.target.id))
                    # any
                    pos_var_name_any = '__reserved_any_pos_{}_{}'.format(
                        iter_var_hash, self.gen_random()
                    )
                    pos_var_any = _ir.PrimVar(pos_var_name_any, _ir.PrimType('int64'))
                    inline_expr_any = _ir.op.object_get_item(span, iter_expr, pos_var_any)
                    alloca_any_var = _ir.AllocaVarStmt(
                        node.target.id, _ir.ObjectType(), inline_expr_any, span
                    )
                    self.context.update_symbol(node.target.id, alloca_any_var.var)
                    forbody_any = self.to_seq_stmt(
                        [alloca_any_var] + self.parse_body(), span
                    )
                    self.context.pop_scope()

                    # nd
                    pos_var_name_nd = '__reserved_nd_pos_{}_{}'.format(
                        iter_var_hash, self.gen_random()
                    )
                    pos_var_nd = _ir.PrimVar(pos_var_name_nd, _ir.PrimType('int64'))
                    inline_expr_nd = _ir.op.object_get_item(span, iter_expr, pos_var_nd)
                    nd_init_expr = _ir.HLOCast(_ir.NDArrayType(), inline_expr_nd, span)
                    alloca_nd_var = _ir.AllocaVarStmt(
                        node.target.id, _ir.NDArrayType(), nd_init_expr, span
                    )
                    self.context.new_scope(nodes=node.body)
                    self.context.update_symbol(node.target.id, alloca_nd_var.var)
                    try:
                        forbody_nd = self.to_seq_stmt(
                            [alloca_nd_var] + self.parse_body(), span
                        )
                    except:
                        forbody_nd = None
                    self.context.pop_scope()

                    # condition
                    nd_dim = _ir.HLOCastPrim("int64", _ir.op.object_dim(span, iter_expr), span)
                    dim_cond = _ir.PrimGT(nd_dim, _ir.const(1, "int64"), span)

                    # for body
                    for_stmt_any = _ir.For(
                        pos_var_any,
                        _ir.const(0, "int64"),
                        _ir.op.object_len(span, iter_expr),
                        _ir.const(1, "int64"),
                        0,
                        1,
                        forbody_any,
                        span)
                    if forbody_nd is None:
                        # iter is not ndarray type
                        if alloca_iter_expr_var is not None:
                            return _ir.SeqStmt([alloca_iter_expr_var, for_stmt_any], span)
                        else:
                            return for_stmt_any
                    for_stmt_nd = _ir.For(
                        pos_var_nd,
                        _ir.const(0, "int64"),
                        _ir.op.object_len(span, iter_expr),
                        _ir.const(1, "int64"),
                        0,
                        1,
                        forbody_nd,
                        span)
                    if_stmt = _ir.IfThenElse(dim_cond, for_stmt_nd, for_stmt_any, span)
                    if alloca_iter_expr_var is not None:
                        return _ir.SeqStmt([alloca_iter_expr_var, if_stmt], span)
                    else:
                        return if_stmt
                else:
                    origin_var_ty = _type_rel.infer_iterator_value_type(iter_expr_type)
                    if not isinstance(iter_expr_type, (_ir.FileType,)):
                        var_is_view, new_var_ty = try_to_view_type(origin_var_ty)
                    else:
                        var_is_view, new_var_ty = False, origin_var_ty
                    var = _ir.HLOVar(node.target.id, new_var_ty)
                    self.context.update_symbol(node.target.id, var)
                    alloc_holder_var = None
                    ref_and_usage = None
                    if var_is_view:
                        holder_name = '{}_holder_{}'.format(
                            node.target.id, self.gen_random()
                        )
                        alloc_holder_var = _ir.AllocaVarStmt(holder_name, origin_var_ty, span=span)
                        ref_and_usage = self.context.bind_reference(var, alloc_holder_var.var)
                    for_stmts = self.parse_body()
                    if var_is_view and ref_and_usage[1] > 0:
                        for_stmts.insert(0, alloc_holder_var)
                        forbody = self.to_seq_stmt(for_stmts, span)
                    else:
                        forbody = self.to_seq_stmt(for_stmts, span)
                    self.context.pop_scope()
                    return _ir.AutoFor([var], iter_expr, forbody, span)
            elif isinstance(node.target, ast.Tuple):
                self.check_loop_vars(node.target, iter_expr)
                unroll = isinstance(iter_expr, (_ir.expr.HLOEnumerate, _ir.expr.HLOZip))
                loop_vars_ty = _type_rel.infer_iterator_value_type(iter_expr_type)
                all_var_infos = []
                loop_vars = []
                for index, elt in enumerate(node.target.elts):
                    if not isinstance(elt, ast.Name):
                        self.report_error(
                            'The target of AutoFor should be a name.',
                            NotImplementedError)
                    symbol = self.context.lookup_symbol(elt.id)
                    if symbol is not None:
                        self.report_error("name '%s' is redefined in AutoFor" % elt.id)
                    origin_var_ty = _type_rel.infer_nth_item_type(loop_vars_ty, index)
                    var_is_view, new_var_ty = False, origin_var_ty
                    if unroll and not isinstance(origin_var_ty, (_ir.FileType,)):
                        var_is_view, new_var_ty = try_to_view_type(origin_var_ty)
                    loop_var = _ir.HLOVar(elt.id, new_var_ty)
                    self.context.update_symbol(elt.id, loop_var)
                    loop_vars.append(loop_var)
                    if var_is_view:
                        holder_name = '{}_holder_{}'.format(
                            elt.id, self.gen_random()
                        )
                        alloc_holder_var = _ir.AllocaVarStmt(holder_name, origin_var_ty, span=span)
                        ref_and_usage = self.context.bind_reference(loop_var, alloc_holder_var.var)
                        all_var_infos.append([alloc_holder_var, ref_and_usage])
                for_stmts = self.parse_body()
                for var_info in all_var_infos:
                    if var_info[1][1] > 0:
                        for_stmts.insert(0, var_info[0])
                forbody = self.to_seq_stmt(for_stmts, span)
                self.context.pop_scope()
                return _ir.AutoFor(loop_vars, iter_expr, forbody, span)
            else:
                self.report_error(
                    'Unsupported AutoFor target: {}'.format(
                        type(
                            node.target)),
                    'Unsupported feature')

    def visit_If(self, node):
        """If visitor
        AST abstract grammar:
            If(expr test, stmt* body, stmt* orelse)
        """

        span = self.build_span(node)
        condition = self.visit(node.test)
        # then body
        self.context.new_scope(nodes=node.body)
        then_body = self.to_seq_stmt(self.parse_body(), span)
        self.context.pop_scope()

        # else body
        if len(node.orelse) > 0:
            self.context.new_scope(nodes=node.orelse)
            else_body = self.to_seq_stmt(self.parse_body(), span)
            self.context.pop_scope()
        else:
            else_body = None

        return _ir.IfThenElse(condition, then_body, else_body, span)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        span = self.build_span(node)
        # body
        self.context.new_scope(nodes=node.body)
        body = self.to_seq_stmt(self.parse_body(), span)
        self.context.pop_scope()
        # warning: type and name is not supported
        return _ir.ExceptionHandler(None, body, span)

    def visit_Try(self, node: ast.Try):
        span = self.build_span(node)

        # body
        self.context.new_scope(nodes=node.body)
        body = self.to_seq_stmt(self.parse_body(), span)
        self.context.pop_scope()

        # handlers
        self.context.new_scope(nodes=node.handlers)
        handlers = self.parse_body()
        self.context.pop_scope()

        # warning: orelse and final is not supported

        return _ir.TryExcept(body, handlers, span)

    def visit_Raise(self, node: ast.Raise):
        span = self.build_span(node)

        # warning: cause is not supported
        assert not node.cause
        if node.exc is None:
            return _ir.Raise(None, span)
        exc = self.visit(node.exc)
        return _ir.Raise(exc, span)

    def visit_While(self, node: ast.While):
        span = self.build_span(node)
        test = self.visit(node.test)
        self.context.new_scope(nodes=node.body)
        forbody = self.to_seq_stmt(self.parse_body(), span)
        self.context.pop_scope()
        return _ir.While(test, forbody, span)

    def visit_Break(self, node: ast.Break):
        return _ir.Break()

    def visit_Continue(self, node: ast.Continue):
        return _ir.Continue()

    def visit_Call(self, node: ast.Call):
        """Call visitor
        AST abstract grammar:
            Call(expr func, expr* args, keyword* keywords)
            keyword = (identifier? arg, expr value)

        By now 2 patterns of Call is supported:
            1. call builtin function
                1.1 list.append(xx)
            2. call extern
        """
        span = self.build_span(node)

        def is_call_self_script(n: ast.Call):
            if isinstance(n.func, ast.Attribute):
                if n.func.attr == 'script':
                    if isinstance(n.func.value, ast.Name):
                        mod = self.custom_ast_node.module.globals.get(
                            n.func.value.id, NAME_NOT_FOUND
                        )
                        if mod is MATX_MODULE and n.args:
                            user_cls = self.visit(n.args[0])
                            if isinstance(user_cls, context.ASTNode):
                                return True
            return False

        def is_call_self_pmap(n: ast.Call):
            if isinstance(n.func, ast.Attribute):
                if n.func.attr in ('pmap', 'pstarmap', 'apply_async'):
                    if isinstance(n.func.value, ast.Name):
                        mod = self.custom_ast_node.module.globals.get(
                            n.func.value.id, NAME_NOT_FOUND
                        )
                        if mod is MATX_MODULE:
                            return True
            return False

        # ignore nested matx.script
        if isinstance(node.func, ast.Call) and is_call_self_script(node.func):
            node.func = node.func.args[0]
        elif is_call_self_script(node):
            return user_function_wrapper(
                self.visit(node.args[0]), self.current_handle_var, span
            )
        _visit_call_func = self._visit_call_func
        self._visit_call_func = True
        func = self.visit(node.func)
        if hasattr(func, "__RAW_TYPE_2_71828182846___"):
            n = ast.Name()
            n.id = func.__RAW_TYPE_2_71828182846___.__name__
            n.ctx = ast.Load()
            func = self.visit(n)
        self._visit_call_func = _visit_call_func
        if isinstance(func, _ir.NoneExpr):
            self.report_error("'NoneType' object is not callable", TypeError)
        if isinstance(func, context.ASTNode):
            func = func.ir_call_wrapper(self.current_handle_var)
        elif isinstance(func, _ir.BaseExpr):
            if _type_rel.is_type_of(func, _ir.adt.ClassType):
                # explicit class type
                dep_node_ctx = self.sc_ctx.get_ast_node_by_class_type(func.checked_type)
                try:
                    func = dep_node_ctx.context.ir_call_attr(span, func, "__call__")
                except BaseException as e:
                    self.report_error(str(e), SyntaxError)
            elif _type_rel.is_type_of(
                    func, (_ir.type.UserDataType, _ir.type.ObjectType)
            ):
                bound_func = func

                def call_wrapper(span, *args, **kwargs):
                    if len(kwargs) != 0:
                        last_arg = _ir.op.make_kwargs_op(span, **kwargs)
                        return _ir.op.object_call(span, bound_func, *args, last_arg)
                    return _ir.op.object_call(span, bound_func, *args)

                func = call_wrapper
            elif not hasattr(func, '__call__'):
                self.report_error("'{}' object is not callable".format(
                    func.py_type_name()), TypeError)
        if hasattr(func, 'arg_types'):
            pos_args = []
            for arg, exp_ty in zip(node.args, func.arg_types.values()):
                self.push_ann_type(exp_ty)
                pos_args.append(self.visit(arg))
                self.pop_ann_type()
            kw_args = []
            for keyword in node.keywords:
                exp_ty = func.arg_types.get(keyword.arg)
                self.push_ann_type(exp_ty)
                kw_args.append(self.visit(keyword))
                self.pop_ann_type()
            kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        else:
            if func is _ir.op.builtins_isinstance:
                if len(node.args) != 2:
                    self.report_error(
                        f"isinstance expected 2 arguments, got {len(node.args)}",
                        TypeError
                    )
                if len(node.keywords) != 0:
                    self.report_error(f"isinstance() takes no keyword arguments", TypeError)
                func_args_1 = node.args[1]
                if isinstance(func_args_1, ast.Tuple):
                    func_args_1 = tuple([self.visit(elt) for elt in func_args_1.elts])
                else:
                    func_args_1 = self.visit(func_args_1)
                pos_args = [self.visit(node.args[0]), func_args_1]
            elif func is Builtin2Op.registrations["str"]:
                visited_result = self.visit(node.args[0])
                if isinstance(visited_result.checked_type, _ir.ClassType):
                    symbol = visited_result
                    dep_node_ctx = self.sc_ctx.get_ast_node_by_class_type(symbol.checked_type)
                    if "__str__" in dep_node_ctx.context.methods:
                        obj_str_func = dep_node_ctx.context.ir_call_attr(span, symbol, "__str__")
                        visited_result = obj_str_func(span)
                    elif "__repr__" in dep_node_ctx.context.methods:
                        obj_repr_func = dep_node_ctx.context.ir_call_attr(span, symbol, "__repr__")
                        visited_result = obj_repr_func(span)
                pos_args = [visited_result] + [self.visit(arg) for arg in node.args[1:]]
            else:
                pos_args = [self.visit(arg) for arg in node.args]
            kw_args = [self.visit(keyword) for keyword in node.keywords]
            kw_args = {
                kw_arg[0]: user_function_wrapper(kw_arg[1], self.current_handle_var, span)
                for kw_arg in kw_args
            }
        pos_args = [user_function_wrapper(arg, self.current_handle_var, span) for arg in pos_args]
        if is_call_self_pmap(node):
            pos_args.append(self.current_handle_var)
        try:
            return func(span, *pos_args, **kw_args)
        except BaseException as e:
            self.report_error(str(e), SyntaxError)

    def visit_Expr(self, node):
        """Expr visitor"""
        if not isinstance(node.value, (ast.Call, ast.Yield, ast.Str)):
            self.report_error('Only Call, Yield and Str are supported', SyntaxError)
        span = self.build_span(node)
        symbol = self.visit(node.value)
        if isinstance(symbol, _ir.Stmt):
            return symbol
        else:
            return _ir.ExprStmt(symbol, span)

    def visit_Yield(self, node):
        self.fn_is_generator = True
        span = self.build_span(node)
        symbol = self.visit(node.value)
        return _ir.HLOYield(symbol, span)

    def visit_BinOp(self, node):
        """BinOp visitor
        AST abstract grammar:
            BinOp(expr left, operator op, expr right)
        """
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not isinstance(node.op, tuple(MATXScriptParser._binop_maker.keys())):
            self.report_error('This operator is not available', NotImplementedError)
        op = MATXScriptParser._binop_maker[type(node.op)]
        span = self.build_span(node)
        return op(lhs, rhs, span)

    def visit_Compare(self, node):
        """Compare visitor
        AST abstract grammar:
            Compare(expr left, expr right, ops=)
        """

        ops = [self.visit(node.left)]
        ops += [self.visit(comparator) for comparator in node.comparators]
        res = []
        span = self.build_span(node)
        for i in range(len(node.ops)):
            lhs = ops[i]
            rhs = ops[i + 1]
            if isinstance(node.ops[i], ast.In):
                func_name = "ir.object_contains"
                op = Builtin2Op.lookup(func_name)
                return op(span, rhs, lhs)
            if isinstance(node.ops[i], ast.NotIn):
                func_name = "ir.object_contains"
                op = Builtin2Op.lookup(func_name)
                return _generic.op_not(op(span, rhs, lhs), span)
            ast_op_node = type(node.ops[i])
            my_binop = MATXScriptParser._binop_maker.get(ast_op_node, None)
            if my_binop is None:
                self.report_error('Unsupported binary operation: {}'.format(ast_op_node.__name__),
                                  NotImplementedError)
            res.append(my_binop(lhs, rhs, span))
        return _ir.op.all(span, *res)

    def visit_BoolOp(self, node):
        """BoolOp visitor
        AST abstract grammar:
            BoolOp(boolop op, expr* values)
        """

        values = [self.visit(value) for value in node.values]
        span = self.build_span(node)
        return MATXScriptParser._boolop_marker[type(node.op)](span, *values)

    def visit_UnaryOp(self, node):
        """UnaryOp visitor
        AST abstract grammar:
            UnaryOp(unaryop op, expr operand)
        """

        operand = self.visit(node.operand)
        if not isinstance(node.op, tuple(MATXScriptParser._unaryop_maker.keys())):
            self.report_error('This operator is not supported', NotImplementedError)
        span = self.build_span(node)
        return MATXScriptParser._unaryop_maker[type(node.op)](operand, span)

    def visit_Subscript(self, node: ast.Subscript):
        """Subscript visitor
        AST abstract grammar:
            Subscript(expr value, slice slice, expr_context ctx)
            slice = Slice(expr? lower, expr? upper, expr? step)
                    | ExtSlice(slice* dims)
                    | Index(expr value)
        By now 2 patterns of Subscript are supported:
            1. Var[index] Buffer element access()
            2. meta[type_key][index], Meta info access
        """
        is_load = True
        if isinstance(node.ctx, ast.Store):
            is_load = False

        symbol = self.visit(node.value)
        if symbol is None:
            if isinstance(node.value, ast.Name):
                self.report_error(
                    f"name '{node.value.id}' was not declared in this scope",
                    SyntaxError
                )
            else:
                self.report_error(f"'{node.value}' was not declared in this scope", SyntaxError)
        if isinstance(symbol, _ir.NoneExpr):
            self.report_error("'NoneType' object is not subscriptable", SyntaxError)

        not_allowed_types = (
            _ir.type.PrimType.IntType,
            _ir.type.PrimType.FloatType,
            _ir.type.PrimType.BoolType,
        )
        if isinstance(symbol, _ir.BaseExpr) and _type_rel.is_type_of(symbol, not_allowed_types):
            self.report_error("'{}' object is not subscriptable".format(
                symbol.py_type_name()), SyntaxError)
        span = self.build_span(node)
        if isinstance(symbol, _ir.expr.BaseExpr):
            if isinstance(node.slice, ast.Slice):
                lower = self.visit(node.slice.lower) if node.slice.lower else _ir.const(0, 'int64')
                upper = self.visit(
                    node.slice.upper) if node.slice.upper else _ir.op.object_len(span, symbol)
                step = self.visit(node.slice.step) if node.slice.step else _ir.const(1, 'int64')
                if isinstance(node.ctx, ast.Load):
                    return _ir.op.object_get_slice(span, symbol, lower, upper, step)
                elif isinstance(node.ctx, ast.Store):
                    lower = self.visit(node.slice.lower) if node.slice.lower else _ir.const(0,
                                                                                            'int64')
                    upper = self.visit(
                        node.slice.upper) if node.slice.upper else _ir.op.object_len(span, symbol)

                    def subscript_assign(rval):
                        return _ir.op.object_set_slice(span, symbol, lower, upper, rval)

                    subscript_assign.checked_type = symbol.checked_type
                    return subscript_assign
                else:
                    raise NotImplementedError("unknown ctx: ", type(node.ctx).__name__)
            elif isinstance(node.slice, ast.Tuple):
                self.report_error(f'[Subscript] node.slice type {node.slice} is not support',
                                  NotImplementedError)
            else:
                if isinstance(node.slice, ast.Index):
                    indexes = self.visit(node.slice.value)
                else:
                    indexes = self.visit(node.slice)

                indexes = list(indexes) if isinstance(indexes, tuple) else [indexes]
                symbol_s = symbol
                for si in range(len(indexes) - 1):
                    symbol_s = _ir.op.object_get_item(span, symbol_s, indexes[si])
                if is_load:
                    return _ir.op.object_get_item(span, symbol_s, indexes[-1])
                else:
                    def subscript_assign(rval):
                        allowed_types = (
                            _ir.type.ListType,
                            _ir.type.DictType,
                            _ir.type.NDArrayType,
                            _ir.type.UserDataType,
                            _ir.type.ObjectType)
                        if not _type_rel.is_type_of(symbol_s, allowed_types):
                            print(symbol_s.checked_type)
                            self.report_error(
                                "'{}' object does not support item assignment".format(
                                    symbol_s.checked_type), SyntaxError)
                        return _ir.op.object_set_item(span, symbol_s, indexes[-1], rval)

                    subscript_assign.checked_type = symbol_s.checked_type
                    return subscript_assign

        else:
            res = symbol[self.visit(slice)]
            if res is None:
                self.report_error('Only var can be subscriptable', SyntaxError)
            return res

    def visit_Attribute(self, node: ast.Attribute):
        """Attribute visitor
        AST abstract grammar:
            Attribute(expr value, identifier attr, expr_context ctx)
        """
        symbol = self.visit(node.value)
        span = self.build_span(node)
        if isinstance(symbol, _ir.NoneExpr):
            self.report_error("'NoneType' object has no attribute {}".format(node.attr),
                              AttributeError)
        not_allowed_types = (
            _ir.type.PrimType.IntType,
            _ir.type.PrimType.FloatType,
            _ir.type.PrimType.BoolType,
        )
        if isinstance(symbol, _ir.BaseExpr) and _type_rel.is_type_of(symbol, not_allowed_types):
            self.report_error("'{}' object has no attribute {}".format(
                symbol.py_type_name(), node.attr), AttributeError)
        if inspect.ismodule(symbol):
            # lookup builtin module: matx unicodedata and so on...
            if not hasattr(symbol, node.attr):
                self.report_error("{} object '{}' has no attribute '{}'".format(
                    type(symbol).__name__,
                    symbol.__name__,
                    node.attr,
                ), AttributeError)
            module_attr = getattr(symbol, node.attr)
            if inspect.ismodule(module_attr):
                return module_attr
            if isinstance(module_attr, (type(None), int, float, bool, str, bytes, bytearray)):
                return _ir.generic_const(module_attr)

            # lookup depend node
            dep_node = self.custom_ast_node.get_dep_by_module_attr(symbol, node.attr)
            if dep_node is not None:
                return dep_node

            # process numpy or torch
            if symbol.__name__ == "numpy":
                def wrapped_op(sp, *args):
                    return _ir.op.numpy_ops(sp, node.attr, *args)

                return wrapped_op
            elif symbol.__name__ == "torch":
                def wrapped_op(sp, *args):
                    return _ir.op.torch_ops(sp, node.attr, *args)

                return wrapped_op

            def op_wrapper(op, name):
                def wrappped_op(span, *args):
                    return op(span, name, *args)

                return wrappped_op

            if inspect.isclass(module_attr) and issubclass(module_attr, NativeClass):
                return op_wrapper(_ir.op.matx_make_native_object, node.attr)
            if isinstance(module_attr, NativeFunction):
                return op_wrapper(_ir.op.matx_call_native_function, node.attr)
            if inspect.isclass(module_attr) and issubclass(module_attr, OpKernel):
                op_cls_name = module_attr.__name__
                plugin_loader = PluginLoader.lookup(op_cls_name)
                if plugin_loader:
                    plugin_loader()

                def wrapped_native_op(span, *args, **kwargs):
                    return _ir.op.matx_make_native_op(span, module_attr, *args, **kwargs)

                return wrapped_native_op
            op = Builtin2Op.lookup(module_attr)
            if op is None:
                op = Builtin2Op.lookup(symbol.__name__ + "." + node.attr)
            if op is None:
                self.report_error('{}()'.format(module_attr), NotImplementedError)
            return op
        elif isinstance(symbol, _ir.BaseExpr):
            if _type_rel.is_type_of(symbol, _ir.adt.ClassType):
                # explicit class type
                dep_node_ctx = self.sc_ctx.get_ast_node_by_class_type(symbol.checked_type)
                try:
                    return dep_node_ctx.context.ir_call_attr(span, symbol, node.attr)
                except BaseException as e:
                    return self.report_error(str(e), SyntaxError)
            else:
                # local var
                if isinstance(self.parent_node, ast.Call) and self._visit_call_func:
                    # function call
                    if isinstance(symbol.checked_type, _ir.UserDataType):
                        def user_data_wrapper(span, *args, **kwargs):
                            return _ir.op.object_call_attr(
                                span, symbol, node.attr, *args, **kwargs
                            )

                        return user_data_wrapper
                    else:
                        is_ft_split = (node.attr == 'split') and self.current_ann_type is not None \
                            and self.current_ann_type.is_full_typed()
                        if is_ft_split:
                            if not isinstance(
                                    symbol.checked_type, (_ir.StringType, _ir.UnicodeType)):
                                self.report_error(
                                    "split with FT type annotation is only supported for bytes/str, but get '{}'".format(
                                        symbol.py_type_name()), TypeError)
                            op = Builtin2Op.lookup("ir.str_" + node.attr + "_ft")
                        else:
                            op = Builtin2Op.lookup("ir.object_" + node.attr)
                        if op is None:
                            if isinstance(symbol.checked_type, _ir.ObjectType):
                                def user_data_wrapper(span, *args, **kwargs):
                                    return _ir.op.object_call_attr(
                                        span, symbol, node.attr, *args, **kwargs)

                                return user_data_wrapper
                            else:
                                self.report_error(
                                    "'{}' object has no attribute {}".format(
                                        symbol.checked_type, node.attr), AttributeError)

                        def op_wrapper(span, *args, **kwargs):
                            # unbound
                            if is_ft_split:
                                item_type = self.current_ann_type.item_type
                                return op(span, symbol, item_type, *args, **kwargs)
                            return op(span, symbol, *args, **kwargs)

                        return op_wrapper
                else:
                    # ud.xx
                    if isinstance(node.ctx, ast.Store):
                        def _set_attr(item):
                            return _ir.op.object_set_attr(span, symbol, node.attr, item)

                        return _set_attr
                    return _ir.op.object_get_attr(span, symbol, node.attr)
        else:
            # direct use python sugar
            if symbol is None:
                self.report_error('Unsupported Attribute expression', NotImplementedError)
            if not hasattr(symbol, node.attr):
                self.report_error("{} object '{}' has no attribute '{}'".format(
                    type(symbol).__name__,
                    symbol.__name__,
                    node.attr,
                ), AttributeError)
            res = getattr(symbol, node.attr)
            return res

    def visit_Dict(self, node: ast.Dict):
        """Dict visitor
        AST abstract grammar:
            Dict(expr* keys, expr* values)
        """
        ret_type = _ir.DictType()
        if isinstance(self.current_ann_type, _ir.DictType):
            ret_type = self.current_ann_type
        span = self.build_span(node)
        keys = []
        for key in node.keys:
            self.push_ann_type(ret_type.key_type)
            keys.append(self.visit(key))
            self.pop_ann_type()
        values = []
        for value in node.values:
            self.push_ann_type(ret_type.value_type)
            values.append(user_function_wrapper(self.visit(value), self.current_handle_var, span))
            self.pop_ann_type()
        init = {key: value for key, value in zip(keys, values)}
        init_expr = _ir.InitializerDict(init, span=span)
        # dict_cons = _ir.Constructor("Dict", ret_type=init_expr.checked_type)
        if ret_type.is_full_typed():
            inputs_ty = [ret_type.key_type, ret_type.value_type]
            dict_cons = _ir.Constructor("FTDict", inputs=inputs_ty, ret_type=ret_type)
        else:
            dict_cons = _ir.Constructor("Dict", ret_type=ret_type)
        return dict_cons(span, init_expr)

    def visit_Tuple(self, node: ast.Tuple):
        """Tuple visitor
        AST abstract grammar:
            Tuple(expr* elts, expr_context ctx)
        """
        span = self.build_span(node)
        exp_fields = None
        if isinstance(self.current_ann_type, _ir.TupleType):
            exp_fields = self.current_ann_type.fields
        fields = []
        if exp_fields is not None:
            assert len(exp_fields) == len(node.elts)
            for elt, exp_ty in zip(node.elts, exp_fields):
                self.push_ann_type(exp_ty)
                fields.append(
                    user_function_wrapper(self.visit(elt), self.current_handle_var, span)
                )
                self.pop_ann_type()
        else:
            fields = [
                user_function_wrapper(self.visit(elt), self.current_handle_var, span)
                for elt in node.elts
            ]
        field_types = [arg.checked_type for arg in fields]
        ret_ty = _ir.TupleType(field_types)
        init_expr = _ir.InitializerList(fields, span=span)
        tuple_cons = _ir.Constructor("Tuple", inputs=field_types, ret_type=ret_ty)
        return tuple_cons(span, init_expr)

    def visit_List(self, node: ast.List):
        """List visitor
        AST abstract grammar:
            List(expr* elts, expr_context ctx)
        """

        span = self.build_span(node)
        ret_type = _ir.ListType()
        if isinstance(self.current_ann_type, _ir.ListType):
            ret_type = self.current_ann_type
        fields = []
        for element in node.elts:
            self.push_ann_type(ret_type.item_type)
            fields.append(user_function_wrapper(self.visit(element), self.current_handle_var, span))
            self.pop_ann_type()
        init_expr = _ir.InitializerList(fields, span=span)
        # list_cons = _ir.Constructor("List", ret_type=init_expr.checked_type)
        if ret_type.is_full_typed():
            list_cons = _ir.Constructor("FTList", inputs=[ret_type.item_type], ret_type=ret_type)
        else:
            list_cons = _ir.Constructor("List", ret_type=ret_type)
        return list_cons(span, init_expr)

    def visit_Set(self, node: ast.Set):
        """Set visitor
        AST abstract grammar:
            Set(expr* elts, expr_context ctx)
        """

        span = self.build_span(node)
        ret_type = _ir.SetType()
        if isinstance(self.current_ann_type, _ir.SetType):
            ret_type = self.current_ann_type
        fields = []
        for element in node.elts:
            self.push_ann_type(ret_type.item_type)
            fields.append(user_function_wrapper(self.visit(element), self.current_handle_var, span))
            self.pop_ann_type()
        init_expr = _ir.InitializerList(fields, span=span)
        item_type = init_expr.checked_type.item_type
        # set_cons = _ir.Constructor("Set", ret_type=_ir.SetType(False, item_type))
        if ret_type.is_full_typed():
            set_cons = _ir.Constructor("FTSet", inputs=[ret_type.item_type], ret_type=ret_type)
        else:
            set_cons = _ir.Constructor("Set", ret_type=ret_type)
        return set_cons(span, init_expr)

    def build_AnyCompLoopStack(self,
                               node: Union[ast.ListComp, ast.DictComp, ast.SetComp],
                               captures: List[_ir.BaseExpr],
                               span: _ir.Span):
        def with_in_captures_or_const(e):
            if isinstance(e, _ir.IntImm):
                return True
            for cap in captures:
                if cap.same_as(e):
                    return True
            return False

        const_expr_1 = _ir.const(1, "int64")
        const_expr_2 = _ir.const(2, "int64")
        reserve_size = const_expr_1
        loop_stack = []
        for gen in node.generators:
            if gen.is_async:
                self.report_error('async is not supported {}', NotImplementedError)
            if (isinstance(gen.iter, ast.Call)
                    and isinstance(gen.iter.func, ast.Name)
                    and gen.iter.func.id in ("range", "xrange")):
                if not isinstance(gen.target, ast.Name):
                    self.report_error("unsupported for range expression", SyntaxError)
                var = _ir.PrimVar(gen.target.id, _ir.PrimType('int64'))
                if len(gen.iter.args) == 1:
                    start = _ir.const(0, 'int64')
                    stop = self.visit(gen.iter.args[0])
                    step = const_expr_1
                elif len(gen.iter.args) == 2:
                    start = self.visit(gen.iter.args[0])
                    stop = self.visit(gen.iter.args[1])
                    step = const_expr_1
                elif len(gen.iter.args) == 3:
                    start = self.visit(gen.iter.args[0])
                    stop = self.visit(gen.iter.args[1])
                    step = self.visit(gen.iter.args[2])
                else:  # error report
                    if len(gen.iter.args) == 0:
                        self.report_error("No arguments supplied in range expression.")
                    else:
                        self.report_error("Invalid arguments in range expression")
                start = _generic._cast_to_prim_int(start, span)
                stop = _generic._cast_to_prim_int(stop, span)
                step = _generic._cast_to_prim_int(step, span)
                self.context.new_scope()
                self.context.update_symbol(gen.target.id, var)
                if_conds = [self.visit(if_n) for if_n in gen.ifs]
                loop_stack.append([0, var, (start, stop, step), if_conds])
                if (with_in_captures_or_const(start)
                        and with_in_captures_or_const(stop)
                        and with_in_captures_or_const(step)):
                    cur_loops = (stop - start) // step
                    cond = cur_loops > _ir.const(0, "int64")
                    reserve_size *= _ir.op.if_then_else(span, cond, cur_loops, const_expr_1)
                else:
                    reserve_size *= const_expr_2
            else:
                iter_expr = self.visit(gen.iter)
                new_var_ty = _type_rel.infer_iterator_value_type(iter_expr.checked_type)
                self.context.new_scope()
                if isinstance(gen.target, ast.Name):
                    if not isinstance(iter_expr.checked_type, _ir.FileType):
                        if isinstance(new_var_ty, _ir.UnicodeType):
                            new_var_ty = _ir.UnicodeType(is_view=True)
                        if isinstance(new_var_ty, _ir.StringType):
                            new_var_ty = _ir.StringType(is_view=True)
                    var = _ir.HLOVar(gen.target.id, new_var_ty)
                    self.context.update_symbol(gen.target.id, var)
                    loop_stack.append([1, [var], iter_expr, [self.visit(if_n) for if_n in gen.ifs]])
                elif isinstance(gen.target, ast.Tuple):
                    self.check_loop_vars(gen.target, iter_expr)
                    loop_vars = []
                    for index, elt in enumerate(gen.target.elts):
                        if not isinstance(elt, ast.Name):
                            self.report_error(
                                'Target of AutoFor should be a name.',
                                'Unsupported feature'
                            )
                        loop_var = _ir.HLOVar(
                            elt.id,
                            _type_rel.infer_nth_item_type(new_var_ty, index)
                        )
                        self.context.update_symbol(elt.id, loop_var)
                        loop_vars.append(loop_var)
                    if_conds = [self.visit(if_n) for if_n in gen.ifs]
                    loop_stack.append([1, loop_vars, iter_expr, if_conds])
                else:
                    self.report_error(
                        'Unsupported AutoFor target: {}'.format(type(gen.target)),
                        'Unsupported feature')
                if isinstance(iter_expr.checked_type, (_ir.ListType,
                                                       _ir.DictType,
                                                       _ir.SetType,
                                                       _ir.StringType,
                                                       _ir.UnicodeType,
                                                       _ir.TupleType,)):
                    if with_in_captures_or_const(iter_expr):
                        reserve_size *= _ir.op.object_len(None, iter_expr)
                    else:
                        reserve_size *= const_expr_2
                else:
                    reserve_size *= const_expr_2
        return loop_stack, reserve_size

    def build_AnyCompNestedFor(self, loop_stack, last_stmt, span):
        for loop_ctx in reversed(loop_stack):
            for if_n in reversed(loop_ctx[3]):
                last_stmt = _ir.IfThenElse(if_n, last_stmt, None, span)
            if loop_ctx[0] == 0:
                # for range
                last_stmt = _ir.For(
                    loop_ctx[1],
                    loop_ctx[2][0],
                    loop_ctx[2][1],
                    loop_ctx[2][2],
                    0,
                    1,
                    last_stmt,
                    span)
            else:
                last_stmt = _ir.AutoFor(loop_ctx[1], loop_ctx[2], last_stmt, span)
            self.context.pop_scope()
        return last_stmt

    def build_FreeVars(self, node: ast.AST):
        init_free_vars = []
        if self.class_instance_var is not None:
            init_free_vars.append(self.class_instance_var)
        elif self.session_handle_var_ctx and self.current_handle_var is not None:
            init_free_vars.append(self.current_handle_var)
        captures = CollectLocalVars(self.context, init_free_vars).run(node)
        return captures

    def visit_ListComp(self, node: ast.ListComp):
        # For xxxComprehension, it is replaced by a for loop. In this case,
        # pre-allocate size of container.
        span = self.build_span(node)
        ret_type = _ir.ListType()
        if isinstance(self.current_ann_type, _ir.ListType):
            ret_type = self.current_ann_type
        captures = self.build_FreeVars(node)
        loop_stack, reserve_size = self.build_AnyCompLoopStack(node, captures, span)
        # alloca var
        self.push_ann_type(ret_type.item_type)
        elt = self.visit(node.elt)
        self.pop_ann_type()
        alloca_stmt = _ir.AllocaVarStmt("__reserved_list_comp_result", ret_type, span=span)
        ret_var = alloca_stmt.var
        reserve_stmt = _ir.ExprStmt(_ir.op.object_reserve(span, ret_var, reserve_size), span)
        last_stmt = _ir.ExprStmt(_ir.op.object_append(span, ret_var, elt), span)
        # make nested loops
        last_stmt = self.build_AnyCompNestedFor(loop_stack, last_stmt, span)
        body = _ir.SeqStmt([
            alloca_stmt,
            reserve_stmt,
            last_stmt,
            _ir.ReturnStmt(ret_var, span)
        ])
        return _ir.LambdaFunction(captures, [], body, ret_type, span=span)(span)

    def visit_SetComp(self, node: ast.SetComp):
        # For xxxComprehension, it is replaced by a for loop. In this case,
        # pre-allocate size of container.
        span = self.build_span(node)
        ret_type = _ir.SetType()
        if isinstance(self.current_ann_type, _ir.SetType):
            ret_type = self.current_ann_type
        captures = self.build_FreeVars(node)
        loop_stack, reserve_size = self.build_AnyCompLoopStack(node, captures, span)
        # alloca var
        self.push_ann_type(ret_type.item_type)
        elt = self.visit(node.elt)
        self.pop_ann_type()
        alloca_stmt = _ir.AllocaVarStmt("__reserved_set_comp_result", ret_type, span=span)
        ret_var = alloca_stmt.var
        reserve_stmt = _ir.ExprStmt(_ir.op.object_reserve(span, ret_var, reserve_size), span)
        last_stmt = _ir.ExprStmt(_ir.op.object_add(span, ret_var, elt), span)
        # make nested loops
        last_stmt = self.build_AnyCompNestedFor(loop_stack, last_stmt, span)
        body = _ir.SeqStmt([
            alloca_stmt,
            reserve_stmt,
            last_stmt,
            _ir.ReturnStmt(ret_var, span)
        ])
        return _ir.LambdaFunction(captures, [], body, ret_type, span=span)(span)

    def visit_DictComp(self, node: ast.DictComp):
        # For xxxComprehension, it is replaced by a for loop. In this case,
        # pre-allocate size of container.
        span = self.build_span(node)
        ret_type = _ir.DictType()
        if isinstance(self.current_ann_type, _ir.DictType):
            ret_type = self.current_ann_type
        captures = self.build_FreeVars(node)
        loop_stack, reserve_size = self.build_AnyCompLoopStack(node, captures, span)
        # alloca var
        self.push_ann_type(ret_type.key_type)
        key = self.visit(node.key)
        self.pop_ann_type()
        self.push_ann_type(ret_type.value_type)
        value = self.visit(node.value)
        self.pop_ann_type()
        alloca_stmt = _ir.AllocaVarStmt("__reserved_dict_comp_result", ret_type, span=span)
        ret_var = alloca_stmt.var
        reserve_stmt = _ir.ExprStmt(_ir.op.object_reserve(span, ret_var, reserve_size), span)
        last_stmt = _ir.ExprStmt(_ir.op.object_set_item(span, ret_var, key, value), span)
        # make nested loops
        last_stmt = self.build_AnyCompNestedFor(loop_stack, last_stmt, span)
        body = _ir.SeqStmt([
            alloca_stmt,
            reserve_stmt,
            last_stmt,
            _ir.ReturnStmt(ret_var, span)
        ])
        return _ir.LambdaFunction(captures, [], body, ret_type, span=span)(span)

    def visit_IfExp(self, node: ast.IfExp):
        span = self.build_span(node)
        cond = self.visit(node.test)
        t = self.visit(node.body)
        f = self.visit(node.orelse)
        return _ir.op.if_then_else(span, cond, t, f)

    def visit_keyword(self, node):
        """Keyword visitor
        AST abstract grammar:
            keyword = (identifier? arg, expr value)
        """

        return node.arg, self.visit(node.value)

    def visit_Name(self, node: ast.Name):
        """Name visitor
        AST abstract grammar:
            Name(identifier id, expr_context ctx)
        """
        name = node.id
        symbol = self.context.lookup_symbol(name)
        if symbol is not None:
            return symbol
        raw_name_or_none = getattr(node, 'raw_id', None)
        if raw_name_or_none is not None:
            name = raw_name_or_none

        # lookup by instance
        global_dep = self.custom_ast_node.module.globals.get(name, NAME_NOT_FOUND)
        if global_dep is not NAME_NOT_FOUND:
            if inspect.ismodule(global_dep):
                return global_dep
            if isinstance(global_dep, (type(None), int, float, bool, str, bytes, bytearray)):
                return _ir.generic_const(global_dep)
            if inspect.isclass(global_dep) and issubclass(global_dep, BaseException):
                exc_cls_name = global_dep.__name__

                def _exception_handler(my_span, *args, **kwargs):
                    return _ir.op.builtins_exception(my_span, exc_cls_name, *args, **kwargs)

                return _exception_handler
            if isinstance(global_dep, OpKernel):
                plugin_loader = PluginLoader.lookup(name)
                if not plugin_loader:
                    self.report_error("%s is not registered" % name)
                plugin_loader()

                def wrapped_native_op(span, *args, **kwargs):
                    return _ir.op.matx_make_native_op(span, global_dep, *args, **kwargs)

                return wrapped_native_op
            dep_node = self.custom_ast_node.get_dep_cls_by_raw_type(global_dep)
            if dep_node is not None:
                return dep_node
            symbol = Builtin2Op.lookup_with_dynamic_type(global_dep, self.current_ann_type)
            if symbol is not None:
                return symbol

        # lookup by name
        dep_node = self.custom_ast_node.get_dep_cls_by_name(name)
        if dep_node is not None:
            return dep_node
        symbol = Builtin2Op.lookup_with_dynamic_type(name, self.current_ann_type)
        if symbol is not None:
            return symbol
        if global_dep is None:
            self.report_error(f"name '{name}' was not declared in this scope", SyntaxError)
        else:
            self.report_error(f"'{global_dep}' is not supported", NotImplementedError)

    # note that after Python3.8, ast.NameConstant, ast.Num, ast.Str are no longer used
    def visit_Constant(self, node):
        if node.value is None:
            return _ir.NoneExpr()
        elif isinstance(node.value, str):
            span = self.build_span(node)
            return _ir.UnicodeImm(node.value, span)
        elif isinstance(node.value, numbers.Number):
            return _ir.const(node.value, _type_infer(node.value).dtype)
        elif isinstance(node.value, bytes):
            span = self.build_span(node)
            return _ir.StringImm(node.value, span)
        else:
            raise NotImplementedError(f'Unsupported value {node.value}')

    def visit_NameConstant(self, node):
        if node.value is None:
            return _ir.NoneExpr()
        return _ir.const(node.value, _type_infer(node.value).dtype)

    def visit_Num(self, node):
        return _ir.const(node.n, _type_infer(node.n).dtype)

    def visit_Str(self, node):
        span = self.build_span(node)
        return _ir.UnicodeImm(node.s, span)

    def visit_Bytes(self, node):
        span = self.build_span(node)
        return _ir.StringImm(node.s, span)

    def visit_Pass(self, node):
        return None

    def visit_NoneType(self, node):
        return _ir.NoneExpr()


def user_function_wrapper(value, resource_handle, span):
    if isinstance(value, context.ASTNode) and inspect.isfunction(value.raw):
        return _make_user_function(value, resource_handle, span)
    if isinstance(value, context.ASTNode) and inspect.isclass(value.raw):
        return _make_user_class_creator(value, resource_handle, span)
    if isinstance(value, context.GetClassAttr):
        return value.as_user_function(resource_handle, span)
    return value


def _make_user_function(value, resource_handle, span):
    func_addr = "(MATXScriptBackendPackedCFunc)" + value.context.unbound_name + "__c_api"
    value = _ir.call_extern(_ir.UserDataType(),
                            "MakeUserFunction",
                            span,
                            _ir.StringImm(value.context.unbound_name),
                            _ir.EnumAttr(func_addr),
                            resource_handle)
    return value


def _make_user_class_creator(value, resource_handle, span):
    from . import rules
    unbound_name = rules.NameRule.get_class_init_wrapper(value.raw.__name__)
    func_addr = "(MATXScriptBackendPackedCFunc)" + unbound_name + "__c_api"
    value = _ir.call_extern(_ir.UserDataType(),
                            "MakeUserFunction",
                            span,
                            _ir.StringImm(value.raw.__name__ + ".__init__"),
                            _ir.EnumAttr(func_addr),
                            resource_handle)
    return value
