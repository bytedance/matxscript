# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement:
# 1) Torczon, L. and Cooper, M. ed., (2012). Ch9 - Data-Flow Analysis.
#    In: Engineering a compiler, 2nd ed. Texas: Elsevier, Inc, pp.495-519.
# 2) Torczon, L. and Cooper, M. ed., (2012). Ch8 - Introduction to optimization.
#    In: Engineering a compiler, 2nd ed. Texas: Elsevier, Inc, pp.445-457.
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
from matx._typed_ast import ast
from collections import deque
from typing import List, Dict, Set, Optional, Union


class Variable:

    def __init__(self, ast_node: Union[ast.Name, ast.arg]):
        assert isinstance(ast_node, (ast.Name, ast.arg))
        self.ast_node = ast_node

    def __repr__(self):
        if isinstance(self.ast_node, ast.arg):
            return f"Argument '{self.ast_node.arg}' in line {self.ast_node.lineno}"
        if isinstance(self.ast_node.ctx, ast.Load):
            return f"Load Variable '{self.name}' in line {self.ast_node.lineno}"
        else:
            assert isinstance(self.ast_node.ctx, ast.Store)
            return f"Store Variable '{self.name}' in line {self.ast_node.lineno}"

    @property
    def name(self):
        if isinstance(self.ast_node, ast.arg):
            return self.ast_node.arg
        else:
            return self.ast_node.id

    @property
    def lineno(self):
        return self.ast_node.lineno


class Block:
    """Block Base Class"""

    def __init__(self,
                 *,
                 name: Optional[str] = None,
                 start_line: Optional[int] = None,
                 end_line: Optional[int] = None,
                 block_end_type: Optional[str] = None,
                 scope=None):
        # basic info
        self.name: Optional[str] = name
        self.start_line: Optional[int] = start_line
        self.end_line: Optional[int] = end_line
        self.block_end_type: Optional[str] = block_end_type
        # links to the next blocks in a control flow graph.
        self.next_block_list = []
        # Links to predecessors in a control flow graph.
        self.prev_block_list = []
        # dominator tree: https://www.cs.rice.edu/~keith/EMBED/dom.pdf
        self.reverse_dominators = set()
        self.immediate_dominators = []  # the forward link in the dominator tree
        self.reverse_immediate_dominator = None  # the immediate parent in the dominator tree
        self.dominance_frontier = []
        # var life info
        self.var_kill = set()  # contains all the vars that are defined in this block
        self.ue_var = set()  # contains all the variables that are from upward
        self.live_out = set()  # contains all the vars that lives on exiting this block
        # reaching definition
        self.reach_def_in: Set[Variable] = set()
        self.reach_def_out: Set[Variable] = set()
        self.reach_def_gen: Set[Variable] = set()
        self.reach_def_kill: Set[Variable] = set()
        # statements or condition in the block.
        self.statements: List[ast.AST] = []
        # a scope block containing this block
        self.scope: Optional[ScopeBlock] = scope

    def __str__(self) -> str:
        s = "Block {} from line {} to {}".format(
            self.name, self.start_line, self.end_line
        )
        return s

    def __repr__(self) -> str:
        txt = "{} with {} next_blocks".format(str(self), len(self.next_block_list))
        if self.statements:
            txt += ", body=["
            txt += ", ".join([ast.dump(node) for node in self.statements])
            txt += "]"
        return txt

    def set_dominators(self, dom_blocks):
        self.reverse_dominators = dom_blocks

    def get_num_of_parents(self):
        return len(self.prev_block_list)

    def recompute_live_out_var(self):
        new_live_out = set()
        for next_block in self.next_block_list:
            new_live_out.update(next_block.ue_var)
            defined_live_out = next_block.live_out & next_block.var_kill
            calculated_lo = next_block.live_out - defined_live_out
            new_live_out.update(calculated_lo)
        if len(new_live_out - self.live_out) == 0:
            return False
        self.live_out = new_live_out
        return True

    def get_code_to_analyse(self):
        for code in self.statements:
            yield code

    def add_statement(self, node: Union[ast.expr, ast.stmt]):
        self.statements.append(node)


class ASTNodeContext:

    def __init__(self, ast_tree: ast.AST):
        self._ast_node_to_parent: Dict[ast.AST, ast.AST] = {}
        for node in ast.walk(ast_tree):
            for child in ast.iter_child_nodes(node):
                self._ast_node_to_parent[child] = node
        self._ast_node_to_block: Dict[ast.AST, Block] = {}
        # for nested class or function def
        self._ast_node_containing_scope: Dict[ast.AST, List[ScopeBlock]] = {}

    def set_block(self, ast_node: ast.AST, block: Block) -> None:
        self._ast_node_to_block[ast_node] = block

    def lookup_block(self, ast_node: ast.AST) -> Optional[Block]:
        block = self._ast_node_to_block.get(ast_node, None)
        return block

    def lookup_scope(self, ast_node: ast.AST) -> Optional[Block]:
        parent_node = self._ast_node_to_parent.get(ast_node, None)
        if parent_node is None:
            return None
        parent_block = self._ast_node_to_block.get(parent_node, None)
        assert parent_block is not None, "internal error"
        if isinstance(parent_block, ScopeBlock):
            return parent_block
        return self.lookup_scope(parent_node)

    def lookup_containing_scope(self, ast_node: ast.AST):
        return self._ast_node_containing_scope.get(ast_node, [])


class BasicBlock(Block):
    """Basic block in a control flow graph.

    Contains a list of statements executed in a program without any control flow.
    """

    def __init__(self,
                 *,
                 name: Optional[str] = None,
                 start_line: Optional[int] = None,
                 end_line: Optional[int] = None,
                 block_end_type: Optional[str] = None):
        super(BasicBlock, self).__init__(
            name=name,
            start_line=start_line,
            end_line=end_line,
            block_end_type=block_end_type,
        )

    @classmethod
    def from_list(cls, stmts: List[Union[ast.expr, ast.stmt]],
                  ctx: ASTNodeContext,
                  **kwargs):
        start_line = stmts[0].lineno
        end_line = stmts[-1].lineno
        c = cls(start_line=start_line, end_line=end_line)
        for stmt in stmts:
            c.add_statement(stmt)
            ctx.set_block(stmt, c)
            parent_block = ctx.lookup_scope(stmt)
            if c.scope is None:
                c.scope = parent_block
            else:
                assert (
                    c.scope == parent_block
                ), "code lineno: {} in the same blocks does not refer to the same scope".format(stmt.lineno)
        for attr, value in kwargs.items():
            setattr(c, attr, value)
        return c


class BlockGraphWalker:
    def __init__(self, root):
        self.root = root
        self.closed_block = set()
        self.queue = deque()

    def walk_bfs(self):
        if not self.root:
            return
        self.queue.append(self.root)
        for block in self._walk_bfs():
            yield block

    def _walk_bfs(self):
        while len(self.queue) != 0:
            block = self.queue.popleft()
            if block:
                yield block
                for next_blk in block.next_block_list:
                    if next_blk not in self.queue and next_blk not in self.closed_block:
                        self.queue.append(next_blk)
                self.closed_block.add(block)


def find_blocks_involved(root, block_list):
    if root not in block_list:
        block_list.append(root)
    block_involved = []
    for block in BlockGraphWalker(root).walk_bfs():
        block_involved.append(block)
    # to preserve the sequence of block_list
    result = [blk for blk in block_list if blk in block_involved]
    return result


class ScopeBlock(Block):
    """Block that defined another scope within.
    def foo(z):     ---> ScopeBlock
        if z:       ---> BasicBlock
            x = 2   ---> BasicBlock
    """

    def __init__(self,
                 *,
                 name: Optional[str] = None,
                 start_line: Optional[int] = None,
                 end_line: Optional[int] = None,
                 ast_node: Optional[ast.AST] = None,
                 block_end_type: Optional[str] = None):
        super(ScopeBlock, self).__init__(
            name=name,
            start_line=start_line,
            end_line=end_line,
            block_end_type=block_end_type,
        )
        self.blocks: List[Block] = []
        # ast node that this block wrapped
        self.ast_node = ast_node

    def fill_dominates(self, ctx: ASTNodeContext):
        """solving the data flow equations
        Dom(n) = {n} | (&=Dom(m)) where m = preds(n)
        """
        all_set = set(self.blocks)
        all_set.add(self)
        for blk in self.blocks:
            blk.set_dominators(all_set)
        self.reverse_dominators = {self}
        changed = True
        while changed:
            changed = False
            for blk in self.blocks:
                if len(blk.prev_block_list) > 0:
                    preds_dom = set.intersection(
                        *(prev.reverse_dominators for prev in blk.prev_block_list)
                    )
                else:
                    preds_dom = set()
                preds_dom.add(blk)
                if blk.reverse_dominators != preds_dom:
                    blk.set_dominators(preds_dom)
                    changed = True
        if self.ast_node:
            containing_scopes = ctx.lookup_containing_scope(self.ast_node)
            for scope in containing_scopes:
                scope.fill_dominates(ctx)

    def fill_immediate_dominators(self, ctx: ASTNodeContext):
        """fill `immediate_dominators` and `reverse_immediate_dominator` of all blocks.
        This essentially build the dominator tree.
        """

        def _find_idom(b, rev_doms):
            queue = deque()
            queue.append(b)
            while len(queue) > 0:
                block = queue.pop()
                _visited.append(block)
                for prev_blk in block.prev_block_list:
                    if prev_blk != blk and prev_blk not in _visited:
                        if prev_blk in rev_doms:
                            return prev_blk
                        else:
                            queue.appendleft(prev_blk)
                    else:
                        continue
            return None

        for blk in self.blocks:
            _visited = []
            idom_blk = _find_idom(blk, blk.reverse_dominators)
            blk.reverse_immediate_dominator = idom_blk
            if idom_blk:
                idom_blk.immediate_dominators.append(blk)
        if self.ast_node:
            containing_scopes = ctx.lookup_containing_scope(self.ast_node)
            for scope in containing_scopes:
                scope.fill_immediate_dominators(ctx)

    def fill_dominance_frontier(self, ctx: ASTNodeContext, block_list):
        block_list = find_blocks_involved(self, block_list)
        for nd in block_list:
            if nd.get_num_of_parents() > 1:
                for pred_node in nd.prev_block_list:
                    runner = pred_node
                    if runner in block_list:
                        while runner is not None and runner != nd.reverse_immediate_dominator:
                            runner.dominance_frontier.append(nd)
                            runner = runner.reverse_immediate_dominator
        if self.ast_node:
            containing_scopes = ctx.lookup_containing_scope(self.ast_node)
            for scope in containing_scopes:
                scope.fill_dominance_frontier(ctx, scope.blocks)


class FunctionLabel(ScopeBlock):
    def __init__(self,
                 *,
                 name: Optional[str] = None,
                 start_line: Optional[int] = None,
                 end_line: Optional[int] = None,
                 ast_node: Optional[ast.AST] = None,
                 block_end_type: Optional[str] = None):
        super(FunctionLabel, self).__init__(
            name=name,
            start_line=start_line,
            end_line=end_line,
            ast_node=ast_node,
            block_end_type=block_end_type,
        )
        self.func_tail = []  # blocks that contain return stmt

    @classmethod
    def from_ast(cls,
                 ast_node: ast.FunctionDef,
                 ctx: ASTNodeContext):
        block = cls(
            name=ast_node.name,
            start_line=ast_node.lineno,
            end_line=ast_node.lineno,
            ast_node=ast_node,
            block_end_type=ast.FunctionDef.__name__,
        )
        ctx.set_block(ast_node, block)
        block.scope = ctx.lookup_scope(ast_node)
        return block
