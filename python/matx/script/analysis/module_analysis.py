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

import builtins
from collections import namedtuple
import inspect
from .. import context


AllVars = namedtuple('AllVars', 'nonlocals globals builtins')


def getallvars(obj):
    """
    Get the mapping of *all* variables to their current values.

    Returns a named tuple of dicts mapping the current nonlocal, global
    and builtin references as seen by the body of the function. A final
    set of unbound names that could not be resolved is also provided.
    """

    # FIXME:
    # `getclosurevars` doesn't capture type annotation, it is the reason that we use frame
    # to get all vars in `f_locals` but not those from func.__closure__ and code.__freevars__.
    # Also, `code.co_names` has the same issue that forbidding us to use `getclosurevars`.
    #
    # Commit fa090dbbabb99859afe701f736731104bce77d8e tried to use `getclosurevars` but failed
    # when trying to get Class definations in locals.
    #
    # This version is rather a redundent implementation to save all vars to module info. Anyone
    # can optimize it?

    frame = inspect.currentframe()
    while frame:
        if frame.f_locals.get(obj.__name__, None) is obj:
            break
        frame = frame.f_back

    if frame is None:
        nonlocal_vars = {}
    else:
        nonlocal_vars = frame.f_locals

    if inspect.isfunction(obj):
        global_ns = obj.__globals__
    elif frame is not None:
        global_ns = frame.f_globals
    else:
        global_ns = {}

    builtin_ns = global_ns.get("__builtins__", builtins.__dict__)
    if inspect.ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__

    return AllVars(nonlocal_vars, global_ns, builtin_ns)


class ModuleAnalysis:

    def __init__(self) -> None:
        self.cls_ctx = None

    def run_impl(self, node: context.ASTNode):
        if node.module is None:
            # parser module info
            node.module = context.ModuleInfo()
            node.module.raw = inspect.getmodule(node.raw)

            global_syms = [o for o in inspect.getmembers(node.module.raw)]
            node.module.globals.update(global_syms)

            node.module.name = node.module.raw.__name__

            all_vars = getallvars(node.raw)
            node.module.globals.update(all_vars.builtins)
            node.module.globals.update(all_vars.nonlocals)
            node.module.globals.update(all_vars.globals)

            try:
                if inspect.isfunction(node.raw):
                    vars = inspect.getclosurevars(node.raw)
                    node.module.globals.update(vars.globals)
                    node.module.globals.update(vars.nonlocals)
                    node.module.globals.update(vars.builtins)
                elif inspect.isclass(node.raw):
                    try:
                        vars = inspect.getclosurevars(node.raw.__init__)
                        node.module.globals.update(vars.globals)
                        node.module.globals.update(vars.nonlocals)
                        node.module.globals.update(vars.builtins)
                    except:
                        pass

                    for func in inspect.getmembers(node.raw, inspect.isfunction):
                        if func.__name__ == '__init__':
                            continue
                        vars = inspect.getclosurevars(func)
                        node.module.globals.update(vars.globals)
                        node.module.globals.update(vars.nonlocals)
                        node.module.globals.update(vars.builtins)

            except:
                pass

            for name, obj in node.module.globals.items():
                try:
                    node.module.imports[name] = inspect.getabsfile(obj)
                except TypeError:
                    pass

    def run(self, sc_ctx: context.ScriptContext):
        self.run_impl(sc_ctx.main_node)
        for dep_node in sc_ctx.deps_node:
            self.run_impl(dep_node)
