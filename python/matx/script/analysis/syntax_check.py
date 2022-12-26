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
from typing import Dict, List, Optional
from matx._typed_ast import ast
from .. import context
from ..reporter import raise_syntax_error, unsupported_syntax_error_table, INVALID_ITERATOR_METHODS


class FunctionSyntaxCheck(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_yield = False
        self.custom_node: Optional[context.ASTNode] = None
        self.current_ast_node: Optional[ast.AST] = None
        self.visit_func_cnt = 0
        self.autofor_varnames = []

    def check_control_flow(self, stmts: List[ast.AST]):
        must_return = False
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                must_return |= self.check_control_flow(
                    stmt.body) and self.check_control_flow(
                    stmt.orelse)
            elif isinstance(stmt, ast.Return):
                must_return = True
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                self.has_yield = True
            elif isinstance(stmt, (ast.For, ast.While)):
                self.check_control_flow(stmt.body)
        return must_return

    def run(self, node: context.ASTNode):
        self.custom_node = node
        self.current_ast_node = node.ast
        return self.run_ast_node(node.ast)

    def visit(self, node: ast.AST):
        """Override method in ast.NodeVisitor"""
        last_cur_node = self.current_ast_node
        self.current_ast_node = node
        astname = node.__class__.__name__
        if astname in unsupported_syntax_error_table:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                unsupported_syntax_error_table[astname])
        method = "visit_" + astname
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)
        self.current_ast_node = last_cur_node
        return visit_res

    def run_ast_node(self, node: ast.FunctionDef, custom_node: context.ASTNode = None):
        if self.custom_node is None and self.current_ast_node is None:
            self.custom_node = custom_node
            self.current_ast_node = node

        if node.args.vararg:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                f"Variable-length arguments '*{node.args.vararg.arg}' is not supported.")
        if node.args.kwarg:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                f"Variable-length keyword arguments '**{node.args.kwarg.arg}' is not supported.")
        if node.args.kwonlyargs:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                f"keyword only arguments is not supported.")
        self.visit_func_cnt = 1
        self.visit(node.args)

        if node.returns is None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Function should be annotated with its return type.')

        if ((isinstance(node.returns, ast.NameConstant) and node.returns.value is None)
                or (isinstance(node.returns, ast.Name) and node.returns.id == 'Any')):
            pass
        else:
            all_paths_end_with_return = self.check_control_flow(node.body)
            ctx = self.custom_node.context
            if isinstance(ctx, context.ClassContext):
                is_abstract = ctx.methods[node.name].is_abstract
            else:
                is_abstract = ctx.is_abstract
            if not is_abstract and not self.has_yield and not all_paths_end_with_return:
                raise_syntax_error(
                    self.custom_node,
                    self.current_ast_node,
                    "The Function return type is not None, but not all control paths return a value.")

        for stmt in node.body:
            self.visit(stmt)
        self.has_yield = False

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and \
           isinstance(node.value.func, ast.Attribute):
            attr_func = node.value.func
            if isinstance(attr_func.value, ast.Name) and \
                    attr_func.value.id in self.autofor_varnames and \
                    attr_func.attr in INVALID_ITERATOR_METHODS:
                raise_syntax_error(
                    self.custom_node,
                    self.current_ast_node,
                    '{} is not allowed to be used inside AutoFor'.format(
                        attr_func.attr))

    def visit_For(self, node: ast.For):
        """ For visitor """
        if node.orelse is not None and len(node.orelse) > 0:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'The else under for is not supported, please change another way.')
        if isinstance(node.iter, ast.Name):
            self.autofor_varnames.append(node.iter.id)
            for stmt in node.body:
                self.visit(stmt)
            self.autofor_varnames.pop()

    def visit_While(self, node: ast.While):
        """ While visitor """
        if node.orelse is not None and len(node.orelse) > 0:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'The else under while is not supported, please change another way.')

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """ Try visitor """
        if node.type is not None or node.name is not None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Catching a specific exception is not supported, please modify to \"except:\"')
        return super().generic_visit(node)

    def visit_Try(self, node: ast.Try):
        """ Try visitor """
        if node.orelse is not None and len(node.orelse) > 0:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'The else under try is not supported, please change another way.')
        if node.finalbody is not None and len(node.finalbody) > 0:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'The final under try is not supported, please change another way.')
        return super().generic_visit(node)

    def visit_Raise(self, node: ast.Raise):
        """ Raise visitor """
        if node.cause is not None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                '"raise from xxx" is not supported, please remove "from xxx".')
        if isinstance(node.exc, ast.Call):
            if not isinstance(node.exc.func, ast.Name):
                raise_syntax_error(
                    self.custom_node,
                    self.current_ast_node,
                    'not supported raise exception type')
            supported_except_types = (
                'Exception', 'RuntimeError', 'TypeError', 'ValueError', 'NotImplementedError',
            )
            if node.exc.func.id not in supported_except_types:
                raise_syntax_error(
                    self.custom_node,
                    self.current_ast_node,
                    f'''Only the following exception types are supported: {supported_except_types}''')
        elif node.exc is None:
            pass
        else:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'raise a var/type is not supported, use raise Exception(...) instead.')
        return super().generic_visit(node)

    def visit_arg(self, node: ast.arg):
        """ Args visit """
        if node.arg == 'self':
            return
        if node.annotation is None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Args should be annotated with its type information.')

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.visit_func_cnt += 1

        if self.visit_func_cnt > 1:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Closure is not supported in MATX4.')
        else:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Internal Error.')


class ClassSyntaxCheck(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_info: Dict[str, bool] = {}
        self.annotated_vars: Dict[str, ast.AST] = {}
        self.custom_node: Optional[context.ASTNode] = None
        self.current_ast_node: Optional[ast.AST] = None

    def reset_class_info(self):
        self.class_info = {
            'has_init': False,
            'has_call': False,
        }

    def run(self, node: context.ASTNode):
        self.custom_node = node
        self.current_ast_node = node.ast
        return self.run_ast_node(node.ast)

    def visit(self, node: ast.AST):
        """Override method in ast.NodeVisitor"""
        self.current_ast_node = node
        astname = node.__class__.__name__
        if astname in unsupported_syntax_error_table:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                unsupported_syntax_error_table[astname])
        method = "visit_" + astname
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def run_ast_node(self, node: ast.AST, custom_node: context.ASTNode = None):
        """ Class visitor """
        if self.custom_node is None and self.current_ast_node is None:
            self.custom_node = custom_node
            self.current_ast_node = node

        self.reset_class_info()

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and stmt.target.id == '__slots__' \
                    or isinstance(stmt, ast.Assign) and stmt.targets[0].id == '__slots__':
                # __slots__ assignment
                if not isinstance(stmt.value, (ast.List, ast.Tuple)):
                    raise_syntax_error(
                        self.custom_node,
                        self.current_ast_node,
                        '__slots__ should be either a list or a tuple.')

                if isinstance(stmt, ast.Assign) and len(stmt.value.elts) == 0:
                    continue  # Skip empty slot def

                if isinstance(stmt, ast.Assign) and len(stmt.value.elts) > 0:
                    raise_syntax_error(
                        self.custom_node,
                        self.current_ast_node,
                        'Please use annotation assignment for __slots__(e.g. `__slots__: Tuple[int] = ["para"]`).')

                if isinstance(stmt, ast.AnnAssign) and len(stmt.value.elts) == 0:
                    raise_syntax_error(
                        self.custom_node,
                        self.current_ast_node,
                        'Please don\'t use annotation assignment for __slots__ with no elements.')

                if len(stmt.value.elts) == 1:
                    if isinstance(stmt.annotation.slice, ast.Index):
                        self.annotated_vars[stmt.value.elts[0].s] = stmt.annotation.slice.value
                    else:
                        self.annotated_vars[stmt.value.elts[0].s] = stmt.annotation.slice
                else:
                    if isinstance(stmt.annotation.slice, ast.Tuple):
                        elts = stmt.annotation.slice.elts
                    elif isinstance(stmt.annotation.slice, ast.Index):
                        elts = stmt.annotation.slice.value.elts
                    else:
                        raise NotImplementedError(f'Unsupported annotation {stmt.annotation.slice}')

                    for value_elem, annotation_elem in zip(stmt.value.elts, elts):
                        self.annotated_vars[value_elem.s] = annotation_elem

            elif isinstance(stmt, ast.FunctionDef):
                self.visit(stmt)

            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str):
                pass
            else:
                raise_syntax_error(self.custom_node, self.current_ast_node,
                                   'Class only support function and slots.')

        if not self.class_info["has_init"]:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Class missing function declaration of "__init__".')

    def check_init_func_start_with_self(self, node: ast.AnnAssign):
        return isinstance(node.target, ast.Attribute) and \
            isinstance(node.target.value, ast.Name) \
            and node.target.value.id == 'self'

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """ Function Visit """
        if len(node.args.args) == 0 or node.args.args[0].arg != 'self':
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Class member functions should have a `self` arg as the first argument.')

        if node.returns is None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Class __init__ functions should be annotated with its return type.')

        if node.name == '__init__':
            self.class_info["has_init"] = True
            self.visit(node.args)

            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and \
                        self.check_init_func_start_with_self(stmt) and \
                        (stmt.target.attr not in self.annotated_vars):
                    self.annotated_vars[stmt.target.attr] = stmt.annotation

                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
                    stmt.targets[0], ast.Attribute) and isinstance(stmt.targets[0].value,
                                                                   ast.Name) and stmt.targets[
                        0].value.id == 'self':
                    if stmt.targets[0].attr not in self.annotated_vars:
                        raise_syntax_error(
                            self.custom_node,
                            stmt,
                            'Class __init__ function members require type annotation.')

        if node.name == '__call__':
            self.class_info['has_call'] = True
        func_synatx_check = FunctionSyntaxCheck()
        func_synatx_check.run_ast_node(node, self.custom_node)

    def visit_arg(self, node: ast.arg):
        """ Args visit """
        if node.arg == 'self':
            return
        if node.annotation is None:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Args should be annotated with its type information.')


class ModuleSyntaxCheck(ast.NodeVisitor):
    def __init__(self):
        self.custom_node: Optional[context.ASTNode] = None
        self.current_ast_node: Optional[ast.AST] = None

    def run(self, node: context.ASTNode) -> str:
        self.custom_node = node
        self.current_ast_node = node.ast

        if len(node.ast.body) == 1:
            if isinstance(node.ast.body[0], ast.ClassDef):
                class_synatx_check = ClassSyntaxCheck()
                class_synatx_check.run_ast_node(node.ast.body[0], self.custom_node)
            elif isinstance(node.ast.body[0], ast.FunctionDef):
                func_synatx_check = FunctionSyntaxCheck()
                func_synatx_check.run_ast_node(node.ast.body[0], self.custom_node)
            else:
                raise_syntax_error(
                    self.custom_node,
                    self.current_ast_node,
                    'Only one-function, one-class source code is allowed.')
        else:
            raise_syntax_error(
                self.custom_node,
                self.current_ast_node,
                'Only one-function, one-class source code is allowed.')


class SyntaxCheck(ast.NodeVisitor):
    def __init__(self) -> None:
        self.sc_ctx: Optional[context.ScriptContext] = None

    def run_impl(self, node: context.ASTNode):
        if isinstance(node.ast, ast.Module):
            module_syntax_check = ModuleSyntaxCheck()
            module_syntax_check.run(node)

        if isinstance(node.ast, ast.ClassDef):
            class_syntax_check = ClassSyntaxCheck()
            class_syntax_check.run(node)

        if isinstance(node.ast, ast.FunctionDef):
            func_syntax_check = FunctionSyntaxCheck()
            func_syntax_check.run(node)

    def run(self, sc_ctx: context.ScriptContext):
        self.sc_ctx = sc_ctx
        main_ctx = self.sc_ctx.main_node.context
        if isinstance(
                main_ctx,
                context.ClassContext) and main_ctx.is_abc and main_ctx.abstract_methods:
            raise_syntax_error(
                self.sc_ctx.main_node,
                self.sc_ctx.main_node.ast,
                "Can't instantiate abstract class {} with abstract methods {}".format(
                    main_ctx.name,
                    ', '.join(
                        main_ctx.abstract_methods)))
        self.run_impl(sc_ctx.main_node)
        for dep_node in sc_ctx.deps_node:
            self.run_impl(dep_node)
