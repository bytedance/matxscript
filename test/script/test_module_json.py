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

import os
from typing import List
from typing import Any
import json
import unittest

import matx

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestJson(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"

    def test_json_loads(self):
        @matx.script
        def json_loads_func(s: str) -> Any:
            return json.loads(s)

        @matx.script
        def json_load_from_list(l: List) -> List:
            ret = []
            for s in l:
                ret.append(json.loads(s))
            return ret

        s = '{"\u6211":[2,3],"\u4f60":{"x":4}}'
        result = json_loads_func(s)
        answer = matx.Dict({"\u6211": [2, 3], "\u4f60": {"x": 4}})
        print(result)
        self.assertEqual(result, answer)

        result = json_load_from_list([s])
        self.assertEqual(result[0], answer)

    def test_json_load(self):
        @matx.script
        def json_load_func(fn: str) -> Any:
            fp = open(fn)
            return json.load(fp)

        test_file = self.data_path + "test.json"
        result = json_load_func(test_file)
        answer = matx.Dict({"\u6211": [2, 3], "\u4f60": {"x": 4}})
        print(result)
        self.assertEqual(result, answer)

    def test_json_dumps(self):
        @matx.script
        def json_dumps_func(obj: Any) -> str:
            return json.dumps(obj)

        obj = matx.Dict({"\u6211": [2, 3], "\u4f60": {"x": 4}})
        result = json_dumps_func(obj)
        self.assertEqual(result, '{"\\u4F60":{"x":4},"\\u6211":[2,3]}')

    def test_consistency(self):
        @matx.script
        def json_loads_func(s: str) -> Any:
            return json.loads(s)

        @matx.script
        def json_dumps_func(obj: Any) -> str:
            return json.dumps(obj)

        d = {"\u6211": [2, 3], "\u4f60": {"x": 4}}
        s = json.dumps(d)
        self.assertEqual(json_loads_func(s), matx.Dict(d))

        s = json_dumps_func(matx.Dict(d))
        self.assertEqual(json.loads(s), d)

    def test_runtime_json(self):
        d = {"\u6211": [2, 3], "\u4f60": {"x": 4}}
        s = json.dumps(matx.Dict(d))
        self.assertEqual(json.loads(s), d)

    def test_fused_json(self):
        d = {"\u6211": [2, 3], "\u4f60": {"x": 4}, "cc": (1, "s")}
        pure_d = {'d': d}
        fused_d = {'d': matx.Dict(d)}

        pure_s = json.dumps(pure_d)
        fused_s = json.dumps(fused_d)
        self.assertEqual(json.loads(pure_s), json.loads(fused_s))

    def test_float_inf(self):
        import json
        json_data = '''
        {
          "k1": Infinity,
          "k2": -Infinity
        }
        '''
        data = json.loads(json_data)
        print(data)

        d = matx.serialize(data)
        print(d)
        d2 = matx.deserialize(d)
        print(d2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
