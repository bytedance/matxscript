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
from typing import List, Dict, Tuple, Any


class TestTypeConverter(unittest.TestCase):

    def test_list_item_to_user_type(self):
        class MyDoc:
            def __init__(self, query: str) -> None:
                self.query: str = query

        def func(doc: MyDoc) -> None:
            print(doc.query)

        def f() -> None:
            l: List[MyDoc] = [MyDoc("haha")]
            func(l[0])

        f()
        matx.script(f)()

    def test_container_pop(self):
        class MyDoc:
            def __init__(self, query: str) -> None:
                self.query: str = query

        def test_list_pop() -> Tuple[str, str]:
            l: List[MyDoc] = [MyDoc("haha0"), MyDoc("haha1")]
            ret = (l.pop(0).query, l.pop().query)
            return ret

        def test_dict_pop() -> Tuple[str, str]:
            l: Dict[str, MyDoc] = {"h": MyDoc("haha"), "h2": MyDoc("haha2")}
            h = l.pop("h").query
            h2 = l.pop("h2", 1).query
            return h, h2

        py_ret = test_list_pop()
        tx_ret = matx.script(test_list_pop)()
        assert py_ret == tx_ret

        py_ret = test_dict_pop()
        tx_ret = matx.script(test_dict_pop)()
        assert py_ret == tx_ret


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
