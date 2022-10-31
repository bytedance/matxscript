# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the ScopeContext is inspired by incubator-tvm.
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


class ScopeContext:

    def __init__(self):
        # scope context
        self.node_stack = []  # AST nodes of scopes
        self.symbols = []  # symbols of scopes
        self.referenced_symbols = []  # symbols of scopes

        # function context
        self.func_params = []  # parameter list of function
        self.func_locals_ = {}  # locals_map of function
        self.func_dict_attr = {}  # func_attr of function
        self.func_var_env_dict = {}  # map from var to env_name
        self.func_ret_type = None

        # unique var name
        self.ssa_ctx = {}  # name: count

    def pop_scope(self):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.referenced_symbols.pop()
        self.node_stack.pop()

    def new_scope(self, nodes=None):
        """Creating a new scope"""
        if nodes is None:
            nodes = []
        self.node_stack.append(list(reversed(nodes)))
        self.symbols.append(dict())
        self.referenced_symbols.append(dict())

    def update_symbol(self, name, symbol):
        """Append a symbol into current scope"""
        self.symbols[-1][name] = symbol
        self.referenced_symbols[-1][symbol] = [None, 0]

    def bind_reference(self, symbol, origin_symbol):
        self.referenced_symbols[-1][symbol] = [origin_symbol, 0]
        return self.referenced_symbols[-1][symbol]

    def remove_symbol(self, name):
        """Remove a symbol"""
        for symbols, view_symbols in zip(reversed(self.symbols), reversed(self.referenced_symbols)):
            if name in symbols:
                view_symbols.pop(symbols[name])
                symbols.pop(name)
                return
        raise RuntimeError(
            "Internal error of matx script parser: no symbol named" + name)

    def lookup_symbol(self, name):
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None

    def lookup_symbol_with_level(self, name):
        """Look up symbol by name"""
        level = -1
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name], level
            level -= 1
        return None, None

    def lookup_referenced_symbol(self, symbol, update_count=True):
        for symbols in reversed(self.referenced_symbols):
            if symbol in symbols:
                if update_count:
                    symbols[symbol][1] += 1
                return symbols[symbol][0]
        return None
