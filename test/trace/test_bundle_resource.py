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
import unittest
import uuid
import matx
from matx import pipeline
from typing import Dict
from typing import Any

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestBundleResource(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestBundleResource_%d/" % uuid.uuid4().int
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_simple_bundle_str_file(self):
        @matx.script
        class LookupTable:
            def __init__(self, loc: str) -> None:
                self.table: Dict[str, str] = {}
                for line in open(loc):
                    v = line.split('\t')
                    self.table[v[0]] = v[1].strip()

            def __call__(self, x: str) -> str:
                return self.table[x]

        test_file = self.data_path + "word2id.txt"
        lookup_op = LookupTable(test_file)
        print(lookup_op('hello'))

        def workflow(x):
            return lookup_op(x)

        save_path = self.work_path + "test_bundle_str_file"
        jit_mod = pipeline.Trace(workflow, "hello")
        jit_mod.save(save_path)
        jit_mod = matx.pipeline.Load(save_path, -1)
        ret = jit_mod.run({"x": "hello"})
        self.assertEqual(ret, "0")

    def test_simple_bundle_str_dir(self):
        @matx.script
        class LookupTable:
            def __init__(self, loc: str) -> None:
                self.table: Dict[str, str] = {}
                file_path = loc + "/" + "word2id.txt"
                for line in open(file_path):
                    v = line.split('\t')
                    self.table[v[0]] = v[1].strip()

            def __call__(self, x: str) -> str:
                return self.table[x]

        test_dir = self.data_path + "for_bundle_dir"
        lookup_op = LookupTable(test_dir)
        print(lookup_op('hello'))

        def workflow(x):
            return lookup_op(x)

        save_path = self.work_path + "test_bundle_str_dir"
        jit_mod = pipeline.Trace(workflow, "hello")
        jit_mod.save(save_path)
        jit_mod = matx.pipeline.Load(save_path, -1)
        ret = jit_mod.run({"x": "hello"})
        self.assertEqual(ret, "0")

    def test_complex_bundle_list(self):
        @matx.script
        class LookupTableWithConfig:

            def __init__(self, configs: Any) -> None:
                self.table: Dict[str, str] = {}
                for line in open(configs[2][0]['dict_fn']['default']):
                    v = line.split('\t')
                    self.table[v[0]] = v[1].strip()

            def __call__(self, x: str) -> str:
                return self.table[x]

        test_file = self.data_path + "word2id.txt"
        test_configs = [1, 2, [{'dict_fn': {'default': test_file}, 'other_param': 345}], 'abc']
        lookup_op = LookupTableWithConfig(test_configs)
        print(lookup_op('hello'))

        def workflow(x):
            return lookup_op(x)

        save_path = self.work_path + "test_complex_bundle_list"
        jit_mod = pipeline.Trace(workflow, "hello")
        jit_mod.save(save_path)

        raw_cwd = os.getcwd()
        os.chdir('../')
        try:
            jit_mod = matx.pipeline.Load(save_path, -1)
            ret = jit_mod.run({"x": "hello"})
            self.assertEqual(ret, "0")
        finally:
            os.chdir(raw_cwd)

    def test_complex_bundle_dict(self):
        @matx.script
        class LookupTableWithConfig:
            def __init__(self, configs: Any) -> None:
                self.table: Dict[str, str] = {}
                for line in open(configs['dict_fn']['default']):
                    v = line.split('\t')
                    self.table[v[0]] = v[1].strip()

            def __call__(self, x: str) -> str:
                return self.table[x]

        test_file = self.data_path + "word2id.txt"
        test_configs = {'dict_fn': {'default': test_file}, 'other_param': 345}
        lookup_op = LookupTableWithConfig(test_configs)
        print(lookup_op('hello'))

        def workflow(x):
            return lookup_op(x)

        save_path = self.work_path + "test_complex_bundle_dict"
        jit_mod = pipeline.Trace(workflow, "hello")
        jit_mod.save(save_path)

        raw_cwd = os.getcwd()
        os.chdir('../')
        try:
            jit_mod = matx.pipeline.Load(save_path, -1)
            ret = jit_mod.run({"x": "hello"})
            self.assertEqual(ret, "0")
        finally:
            os.chdir(raw_cwd)

    def test_complex_bundle_multi_paths(self):
        tmp_path = self.work_path + "multi_paths_input"
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        loc_1 = tmp_path + '/test_rss.txt'
        loc_2 = tmp_path + '/test_rss2.txt'
        of = open(loc_1, 'w')
        print('1 2\n2 3', file=of)
        of.close()
        of = open(loc_2, 'w')
        print('4 5\n6 7', file=of)
        of.close()

        test_config = {
            'root': {
                'locations': {
                    'loc_1': loc_1,
                    'loc_2': loc_2,
                }
            }
        }

        @matx.script
        class LookupTableWithConfig:
            def __init__(self, config: Dict) -> None:
                self.table: Dict = {}
                for line in open(config['root']['locations']['loc_1']):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    a, b = line.split()
                    self.table[a] = b
                for line in open(config['root']['locations']['loc_2']):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    a, b = line.split()
                    self.table[a] = b

            def __call__(self, t: Any) -> Any:
                return self.table[t]

        lookup_op = LookupTableWithConfig(test_config)
        save_path = self.work_path + "test_complex_bundle_multi_paths"

        def workflow(t):
            return lookup_op(t)

        mod = matx.pipeline.Trace(workflow, '2')
        mod.save(save_path)

        os.remove(loc_1)
        os.remove(loc_2)

        os.chdir('..')
        mod = matx.pipeline.Load(save_path, -1)
        self.assertEqual(mod.run({'t': '2'}), '3')


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
