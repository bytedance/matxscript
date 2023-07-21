#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import ast
from itertools import chain
from typing import List, TYPE_CHECKING

import matx.kernel.graphIR as _gir
import matx.kernel.typing
from matx import ir as _ir
from matx.ir import generic as _generic
from matx.kernel.typing.broadcast import broadcast as typing_broadcast

if TYPE_CHECKING:
    from matx.kernel.graphIR import Tensor, IntVar

arithmetic_binop_maker = {
    ast.Add: lambda lhs, rhs, span: _generic.add(lhs, rhs, span),
    ast.Sub: lambda lhs, rhs, span: _generic.subtract(lhs, rhs, span),
    ast.Mult: lambda lhs, rhs, span: _generic.multiply(lhs, rhs, span),
    ast.Div: lambda lhs, rhs, span: _generic.divide(lhs, rhs, span),
    ast.FloorDiv: lambda lhs, rhs, span: _generic.floordiv(lhs, rhs, span),
    # ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
    # quick fix for mod sign issue
    ast.Mod: lambda lhs, rhs, span: _generic.floormod(_generic.add(_generic.floormod(lhs, rhs, span), rhs, span), rhs,
                                                      span),
    ast.BitOr: lambda lhs, rhs, span: _generic.bitwise_or(lhs, rhs, span),
    ast.BitAnd: lambda lhs, rhs, span: _generic.bitwise_and(lhs, rhs, span),
    ast.BitXor: lambda lhs, rhs, span: _generic.bitwise_xor(lhs, rhs, span),
    # ast.LShift: lambda lhs, rhs, span: _generic.left_shift(lhs, rhs, span),
    # ast.RShift: lambda lhs, rhs, span: _generic.right_shift(lhs, rhs, span),
}

unaryop_maker = {
    ast.USub: lambda operand, span: _generic.multiply(operand, _ir.const(-1), span),
    ast.Invert: lambda operand, span: _generic.bitwise_not(operand, span),
    ast.Not: lambda operand, span: _generic.op_not(operand, span)
}

boolop_marker = {
    ast.Gt: lambda lhs, rhs, span: _generic.greater_than(lhs, rhs, span),
    ast.GtE: lambda lhs, rhs, span: _generic.greater_or_equal(lhs, rhs, span),
    ast.Lt: lambda lhs, rhs, span: _generic.less_than(lhs, rhs, span),
    ast.LtE: lambda lhs, rhs, span: _generic.less_or_equal(lhs, rhs, span),
    ast.Eq: lambda lhs, rhs, span: _generic.equal(lhs, rhs, span),
    ast.NotEq: lambda lhs, rhs, span: _generic.notequal(lhs, rhs, span),
    ast.Is: lambda lhs, rhs, span: _generic.op_is(lhs, rhs, span),
    ast.IsNot: lambda lhs, rhs, span: _generic.op_not(_generic.op_is(lhs, rhs, span), span),

    ast.And: lambda lhs, rhs, span: _generic.op_and(span, lhs, rhs),
    ast.Or: lambda lhs, rhs, span: _generic.op_or(span, lhs, rhs)
}


def is_graph_ir_scalar(t):
    if isinstance(t, _gir.Scalar):
        return True
    if isinstance(t, _gir.Tensor):
        return is_graph_ir_scalar_shape(t.shape())
    return False


def is_graph_ir_scalar_shape(s):
    return len(s) == 0


def is_compatible(lhs: 'Tensor', rhs: 'Tensor'):
    return lhs.dtype() == rhs.dtype() and lhs.shape() == rhs.shape()


def unwrap_shape(shape: List['IntVar']) -> List:
    unwrapped_shape = []
    for e in shape:
        if isinstance(e, _gir.IntImm):
            unwrapped_shape.append(e.value())
        else:
            unwrapped_shape.append(e.symbolic_value())
    return unwrapped_shape


def broadcast(arr1_shape: List['IntVar'], arr2_shape: List['IntVar']):
    unwrapped_shape1 = unwrap_shape(arr1_shape)
    unwrapped_shape2 = unwrap_shape(arr2_shape)
    var_map = {}
    for origin, unwrapped in zip(chain(arr1_shape, arr2_shape),
                                 chain(unwrapped_shape1, unwrapped_shape2)):
        if unwrapped not in var_map:
            var_map[unwrapped] = origin

    if len(unwrapped_shape1) == 0:
        unwrapped_shape1 = [1]
    if len(unwrapped_shape2) == 0:
        unwrapped_shape2 = [1]

    result_shape, lhs_new_shape, rhs_new_shape = typing_broadcast(
        unwrapped_shape1, unwrapped_shape2)

    def convert_back(shape):
        if len(shape) == 1 and shape[0] == 1:
            return []
        print(shape)
        return list(map(lambda x: var_map[x], shape))

    result_shape = convert_back(result_shape)
    # lhs_new_shape = convert_back(lhs_new_shape)
    # rhs_new_shape = convert_back(rhs_new_shape)
    return result_shape


def convert_to_kernel_type(node: '_gir.Tensor'):
    dtype = node.dtype()
    shape = []
    for s in node.shape():
        if isinstance(s, _gir.IntVar):
            shape.append(s.symbolic_value())
        elif isinstance(s, _gir.IntImm):
            shape.append(s.value())
        else:
            raise SyntaxError(f"not supported shape value {s}")

    if len(shape) == 0:
        shape = [1]
    return matx.kernel.typing.STR_TO_KERNEL_TYPE[dtype][shape]


def draw_graph(graph_nodes):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    def str_node(node):
        if _gir.utils.is_graph_ir_scalar(node):
            return f"scalar({node.name() if len(node.name()) != 0 else f'tmp{id(node)}'})"
        elif isinstance(node, _gir.Tensor):
            return f"Tensor({node.name() if len(node.name()) != 0 else f'tmp{id(node)}'})"
        elif isinstance(node, _gir.IntImm):
            return f"IntImm({node.value()})"
        elif isinstance(node, _gir.IntVar):
            return f"IntVar({node.symbolic_value()})"
        elif isinstance(node, _gir.Operator):
            return f"op({str(type(node)).split('.')[-1][:-2]}({id(node)}))"
        else:
            return f"{type(node)}({id(node)})"

    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in graph_nodes:
        G.add_node(str_node(node))
        if isinstance(node, _gir.CopyOperator):
            G.add_edge(str_node(node.copy_from), str_node(node))
        elif isinstance(node, _gir.Operator):
            for i in node._attrs["inputs"]:
                G.add_edge(str_node(i), str_node(node))
        elif isinstance(node, _gir.Tensor):
            for src in node.src_ops():
                G.add_edge(str_node(src), str_node(node))
            for dst in node.dst_ops():
                G.add_edge(str_node(node), str_node(dst))
        elif isinstance(node, _gir.IntVar):
            ...
        else:
            ...

    try:
        draw_topological_graph(G)
    except nx.NetworkXUnfeasible:
        draw_spring_layout_graph(G)


def draw_spring_layout_graph(G):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, ax=ax, with_labels=True, font_size=8)
    fig.tight_layout()
    plt.show()


def draw_topological_graph(G):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # topological sort
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Draw the graph
    fig, ax = plt.subplots()
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw_networkx(G, pos, ax=ax, with_labels=True, font_size=8)
    fig.tight_layout()
    plt.show()
