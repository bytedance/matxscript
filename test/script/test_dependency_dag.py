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
import unittest
import matx
from typing import Tuple


def global_func() -> str:
    return 'global_func'


class GlobalRss:

    def __init__(self) -> None:
        self.name: str = 'GlobalRss'


class A:
    def __init__(self) -> None:
        pass

    def func(self) -> GlobalRss:
        rss = GlobalRss()
        return rss


class TestDependencyDag(unittest.TestCase):
    def test_namespace(self):
        def inner_func() -> str:
            return 'inner_func'

        class InnerRss:
            def __init__(self, grss: GlobalRss) -> None:
                self.name: str = 'InnerRss'
                self.grss: GlobalRss = grss

        class InnerOP:

            def __init__(self, irss: InnerRss) -> None:
                self.irss: InnerRss = irss

            def __call__(self) -> str:
                parts = matx.List([])
                parts.append(inner_func())
                parts.append(global_func())
                parts.append(self.irss.name)
                parts.append(self.irss.grss.name)
                return ','.join(parts)

        ScriptedOP = matx.script(InnerOP)
        matx.script(inner_func)
        grss = matx.script(GlobalRss)()
        irss = matx.script(InnerRss)(grss)
        op = ScriptedOP(irss)
        self.assertEqual(op(), 'inner_func,global_func,InnerRss,GlobalRss')

    def test_returns(self):
        @matx.script
        def wrapper() -> str:
            a = A()
            return a.func().name

        self.assertEqual(wrapper(), 'GlobalRss')

    def test_init_and_call_attr(self):
        class MyFoo:
            def __init__(self) -> None:
                pass

            def foo(self) -> str:
                return 'MyFoo'

        @matx.script
        def wrapper() -> str:
            return MyFoo().foo()

        self.assertEqual(wrapper(), 'MyFoo')

    def test_shadow_names(self):
        from pydoc import doc

        def ss():
            print("This function can't be compiled.")

        def func(doc: int, ss: int) -> int:
            return doc * ss

        self.assertEqual(matx.script(func)(3, 4), func(3, 4))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
