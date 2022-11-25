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
import os
import matx


class TestJsonSerialize(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(dir_path, '../data/serialize_data.json')

    def test_serialize(self):
        # test with frequently used data structures in serving.
        assert '5' in matx.serialize(5)
        assert "hello" in matx.serialize("hello")
        assert "3.14" in matx.serialize(3.14)
        feed_dict = {"key1": matx.NDArray([0.5] *
                                          8, [2, 4], dtype='float32'), "key2": ["hello", "world"]}
        serialized = matx.serialize(feed_dict)
        matx_feed_dict = matx.deserialize(serialized)
        assert len(matx_feed_dict.keys()) == 2
        assert matx_feed_dict['key1'].shape()[0] == 2
        assert len(matx_feed_dict['key2']) == 2

        # write to file as test data.
        # with open(self.filepath, 'w') as fw:
        #    fw.write(serialized)

    def test_script_serialize(self):
        @matx.script
        def serialize(o: object) -> str:
            return matx.serialize(o)

        @matx.script
        def deserialize(s: str) -> object:
            return matx.deserialize(s)

        # test with frequently used data structures in serving.
        assert '5' in serialize(5)
        assert "hello" in serialize("hello")
        assert "3.14" in serialize(3.14)
        feed_dict = {"key1": matx.NDArray([0.5] *
                                          8, [2, 4], dtype='float32'), "key2": ["hello", "world"]}
        serialized = serialize(feed_dict)
        matx_feed_dict = deserialize(serialized)
        assert serialized == matx.serialize(feed_dict)
        assert matx_feed_dict == feed_dict

    def test_stable_api(self):
        with open(self.filepath, 'r') as fr:
            data = fr.read()
        matx_feed_dict = matx.deserialize(data)
        assert len(matx_feed_dict.keys()) == 2
        assert matx_feed_dict['key1'].shape()[0] == 2
        assert len(matx_feed_dict['key2']) == 2


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
