import ast
import inspect

from typing import Any


def parse_ast(func):
    src_code = inspect.getsource(func)
    src_file_path = inspect.getfile(func)
    source_file_content, src_line_number = inspect.findsource(func)
    src_ast = ast.parse(src_code)
    ast.increment_lineno(src_ast, src_line_number)

    return src_ast, src_file_path, src_line_number, src_code


class KernelParser:

    def __init__(self, func):
        self.func = func
        # get args
        self.signature = inspect.signature(func)
        self.args_types = {k: v.annotation for k, v in self.signature.parameters.items()}
        # todo support method
        # todo update default args
        self.return_types = self.signature.return_annotation
        self.symbols = []
        self.func_name = func.__name__
        self.ast_visitor = KernelNodeVisitor(self.args_types, self.return_types, self.symbols)

    def parse(self):
        src_ast, _, _, _ = parse_ast(self.func)


class KernelNodeVisitor(ast.NodeVisitor):
    def __init__(self, args_types, return_types, symbols):
        self.args_types = args_types
        self.return_types = return_types
        self.symbols = symbols

    def visit_Constant(self, node: ast.Constant) -> Any:
        print("visit_Constant")
        return node.value

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        raise NotImplementedError("visit_FormattedValue is not Implemented")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        raise NotImplementedError("visit_JoinedStr is not Implemented")

    def visit_List(self, node: ast.List) -> Any:
        raise NotImplementedError("visit_List is not Implemented")

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        raise NotImplementedError("visit_Tuple is not Implemented")

    def visit_Set(self, node: ast.Set) -> Any:
        raise NotImplementedError("visit_Set is not Implemented")

    def visit_Dict(self, node: ast.Dict) -> Any:
        raise NotImplementedError("visit_Dict is not Implemented")

    # variables
    def visit_Name(self, node: ast.Name) -> Any:
        print("visit_Name", node.id)
        return node.id

    def visit_Load(self, node: ast.Load) -> Any:
        raise NotImplementedError("visit_Load is not Implemented")

    def visit_Store(self, node: ast.Store) -> Any:
        raise NotImplementedError("visit_Store is not Implemented")

    def visit_Del(self, node: ast.Del) -> Any:
        raise NotImplementedError("visit_Del is not Implemented")

    def visit_Starred(self, node: ast.Starred) -> Any:
        raise NotImplementedError("visit_Starred is not Implemented")

    # Expressions

    def visit_Expr(self, node: ast.Expr) -> Any:
        print("visit_Expr")
        return self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        pass

        # todo finish

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        raise NotImplementedError("visit_BoolOp is not Implemented")

    def visit_Compare(self, node: ast.Compare) -> Any:
        raise NotImplementedError("visit_Compare is not Implemented")

    def visit_Call(self, node: ast.Call) -> Any:
        raise NotImplementedError("visit_Call is not Implemented")

    def visit_keyword(self, node: ast.keyword) -> Any:
        raise NotImplementedError("visit_keyword is not Implemented")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        raise NotImplementedError("visit_IfExp is not Implemented")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        attr_name = node.attr
        if not isinstance(value, list):
            return [value, attr_name]
        return [*value, attr_name]

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Any:
        raise NotImplementedError("visit_NamedExpr is not Implemented")

    # Subscripting
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise NotImplementedError("visit_Subscript is not Implemented")

    def visit_Slice(self, node: ast.Slice) -> Any:
        raise NotImplementedError("visit_Slice is not Implemented")

    # Comprehensions
    def visit_ListComp(self, node: ast.ListComp) -> Any:
        raise NotImplementedError("visit_ListComp is not Implemented")

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        raise NotImplementedError("visit_SetComp is not Implemented")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        raise NotImplementedError("visit_GeneratorExp is not Implemented")

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        raise NotImplementedError("visit_DictComp is not Implemented")

    def visit_comprehension(self, node: ast.comprehension) -> Any:
        raise NotImplementedError("visit_comprehension is not Implemented")

    # statement
    def visit_Assign(self, node: ast.Assign) -> Any:
        assert len(node.targets) == 1, "assigning multiple var at the same time is not supported"
        target = node.targets[0]
        id = self.visit(target)
        value = self.visit(node.value)

        if isinstance(target, ast.Attribute) and id[0] not in self.scope_arrays:
            raise SyntaxError(f"assigning value to variable {id[0]} which is not in this scope")

        if isinstance(target, ast.Name):
            pass
        raise NotImplementedError("visit_Assign is not Implemented")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        pass

    # control flow
    # pattern matching
    # function and class definitions
    # Async and await

    def visit_Return(self, node: ast.Return) -> Any:
        # todo check none return?
        rt_expr = self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        pass

    def _get_type(self, operand):
        if isinstance(operand, str) and operand in self.sdfg.arrays:
            result = type(self.sdfg.arrays[operand])
        elif isinstance(operand, str) and operand in self.scope_arrays:
            result = type(self.scope_arrays[operand])
        # elif isinstance(operand, tuple(dtypes.DTYPE_TO_TYPECLASS.keys())):
        #    if isinstance(operand, (bool, numpy.bool_)):
        #        result.append((operand, 'BoolConstant'))
        #    else:
        #        result.append((operand, 'NumConstant'))
        # elif isinstance(operand, sympy.Basic):
        #    result.append((operand, 'symbol'))
        else:
            result = type(operand)

        return result
