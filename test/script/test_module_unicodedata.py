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
import unicodedata

import matx


class TestUnicodedataModule(unittest.TestCase):

    def setUp(self) -> None:
        self.test_data = "Spicy Jalapen\u0303o"

    def test_unicodedata_normalize(self):
        @matx.script
        def unicodedata_normalize(query: str) -> str:
            return unicodedata.normalize('NFKC', query)

        result = unicodedata_normalize(self.test_data)
        self.assertEqual(result, unicodedata.normalize('NFKC', self.test_data))

        @matx.script
        def unicodedata_normalize_form(form: str, query: str) -> str:
            return unicodedata.normalize(form, query)

        result = unicodedata_normalize_form('NFKC', self.test_data)
        self.assertEqual(result, unicodedata.normalize('NFKC', self.test_data))

    def test_unicode_category(self):
        import unicodedata

        def unicodedate_category(query: str) -> str:
            return unicodedata.category(query)

        result = unicodedate_category("A")
        matx_script_result = matx.script(unicodedate_category)("A")
        self.assertEqual(result, matx_script_result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
