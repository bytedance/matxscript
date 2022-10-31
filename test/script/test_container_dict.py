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
from typing import Dict, List, Tuple, Any
import matx


def catch_raise(func, *args):
    flag = False
    try:
        func(*args)
    except:
        flag = True
    return flag


class TestContainerDict(unittest.TestCase):
    def test_builtin_dict(self):
        @matx.script
        def dict_set_item(container: Dict, key: Any, val: Any) -> Dict:
            container[key] = val
            return container

        @matx.script
        def generic_set_item(container: Any, key: Any, val: Any) -> Any:
            container[key] = val
            return container

        test_data = {"init_k": "init_v"}
        self.assertEqual(dict_set_item(test_data, 10, 10), {"init_k": "init_v", 10: 10})
        self.assertEqual(generic_set_item(test_data, 10, 10),
                         {"init_k": "init_v", 10: 10})

    def test_dict_clear(self):

        @matx.script
        def dict_clear(a: Dict) -> int:
            a.clear()
            return len(a)

        @matx.script
        def generic_clear(a: Any) -> int:
            a.clear()
            return len(a)

        self.assertEqual(dict_clear({'a': 1, 'b': 2}), 0)
        self.assertEqual(generic_clear({'a': 1, 'b': 2}), 0)

    def test_dict_in(self):

        @matx.script
        def dict_in(a: Dict) -> bool:
            if 'c' in a:
                return True
            else:
                return False

        @matx.script
        def generic_in(a: Any) -> bool:
            if 'c' in a:
                return True
            else:
                return False

        self.assertEqual(dict_in({'a': 1, 'b': 2}), False)
        self.assertEqual(dict_in({'a': 1, 'c': 3}), True)
        self.assertEqual(generic_in({'a': 1, 'b': 2}), False)
        self.assertEqual(generic_in({'a': 1, 'c': 3}), True)

    def test_dict_notin(self):
        def dict_notin(a: Dict) -> bool:
            if 'c' not in a:
                return True
            else:
                return False

        def general_notin(a: Any) -> bool:
            if 'c' not in a:
                return True
            else:
                return False

        dict_notin_op = matx.script(dict_notin)
        self.assertEqual(dict_notin({'a': 1, 'b': 2}), True)
        self.assertEqual(dict_notin({'a': 1, 'c': 3}), False)
        self.assertEqual(dict_notin_op({'a': 1, 'b': 2}), True)
        self.assertEqual(dict_notin_op({'a': 1, 'c': 3}), False)

        general_notin_op = matx.script(general_notin)
        self.assertEqual(general_notin({'a': 1, 'b': 2}), True)
        self.assertEqual(general_notin({'a': 1, 'c': 3}), False)
        self.assertEqual(general_notin_op({'a': 1, 'b': 2}), True)
        self.assertEqual(general_notin_op({'a': 1, 'c': 3}), False)

    def test_dict_reserve(self):

        @matx.script
        def dict_reserve(a: matx.Dict, new_size: int) -> matx.Dict:
            a.reserve(new_size)
            return a

        a = matx.Dict()
        dict_reserve(a, 100)
        self.assertGreaterEqual(a.bucket_count(), 100)
        dict_reserve(a, 50)
        self.assertGreaterEqual(a.bucket_count(), 50)
        dict_reserve(a, -1)
        self.assertGreaterEqual(a.bucket_count(), 50)

    def test_general_reserve(self):

        @matx.script
        def general_reserve(a: Any, new_size: int) -> Any:
            a.reserve(new_size)
            return a

        a = matx.Dict()
        general_reserve(a, 100)
        self.assertGreaterEqual(a.bucket_count(), 100)
        general_reserve(a, 50)
        self.assertGreaterEqual(a.bucket_count(), 50)
        general_reserve(a, -1)
        self.assertGreaterEqual(a.bucket_count(), 50)

    def test_dict_items(self):
        def dict_items(a: Dict) -> Tuple[List, List]:
            k_list = []
            v_list = []
            for item in a.items():
                k, v = item
                k_list.append(k)
                v_list.append(v)
            return k_list, v_list

        dict_items_op = matx.script(dict_items)

        d = {'a': 1, 'b': 2, 'c': 3}
        key_list, value_list = dict_items(d)
        script_key_list, script_value_list = dict_items_op(d)
        self.assertCountEqual(['a', 'b', 'c'], key_list)
        self.assertCountEqual([1, 2, 3], value_list)
        self.assertCountEqual(['a', 'b', 'c'], script_key_list)
        self.assertCountEqual([1, 2, 3], script_value_list)

    def test_general_items(self):
        def general_items(a: Any) -> Tuple[List, List]:
            k_list = []
            v_list = []
            for item in a.items():
                k, v = item
                k_list.append(k)
                v_list.append(v)
            return k_list, v_list

        general_items_op = matx.script(general_items)

        d = {'a': 1, 'b': 2, 'c': 3}
        key_list, value_list = general_items(d)
        script_key_list, script_value_list = general_items_op(d)
        self.assertCountEqual(['a', 'b', 'c'], key_list)
        self.assertCountEqual(['a', 'b', 'c'], script_key_list)
        self.assertCountEqual([1, 2, 3], value_list)
        self.assertCountEqual([1, 2, 3], script_value_list)

    def test_dict_keys(self):
        def dict_keys(a: Dict) -> List:
            k_list = []
            for k in a.keys():
                k_list.append(k)
            return k_list

        dict_keys_op = matx.script(dict_keys)

        d = {'a': 1, 'b': 2, 'c': 3}
        keys_ret = dict_keys(d)
        keys_op_ret = dict_keys_op(d)
        self.assertCountEqual(['a', 'b', 'c'], keys_ret)
        self.assertCountEqual(['a', 'b', 'c'], keys_op_ret)

    def test_general_keys(self):
        def general_keys(a: Any) -> List:
            k_list = []
            for k in a.keys():
                k_list.append(k)
            return k_list

        general_keys_op = matx.script(general_keys)

        d = {'a': 1, 'b': 2, 'c': 3}
        keys_ret = general_keys(d)
        keys_op_ret = general_keys_op(d)
        self.assertCountEqual(['a', 'b', 'c'], keys_ret)
        self.assertCountEqual(['a', 'b', 'c'], keys_op_ret)

    def test_dict_values(self):
        def dict_values(a: Dict) -> List:
            v_list = []
            for v in a.values():
                v_list.append(v)
            return v_list

        dict_values_op = matx.script(dict_values)

        d = {'a': 1, 'b': 2, 'c': 3}
        values_ret = dict_values(d)
        values_op_ret = dict_values_op(d)
        self.assertCountEqual([1, 2, 3], values_ret)
        self.assertCountEqual([1, 2, 3], values_op_ret)

    def test_general_values(self):
        def general_values(a: Any) -> List:
            v_list = []
            for v in a.values():
                v_list.append(v)
            return v_list

        general_values_op = matx.script(general_values)

        d = {'a': 1, 'b': 2, 'c': 3}
        values_ret = general_values(d)
        values_op_ret = general_values_op(d)
        self.assertCountEqual([1, 2, 3], values_ret)
        self.assertCountEqual([1, 2, 3], values_op_ret)

    def test_dict_get(self):
        def dict_get(a: Dict) -> List:
            r = []
            r.append(a.get('known'))
            r.append(a.get('known', 10000))
            r.append(a.get('unknown'))
            r.append(a.get('unknown', 10000))
            r.append(a.get('known', [1, 2, 3]))
            r.append(a.get('unknown', [1, 2, 3]))
            return r

        dict_get_op = matx.script(dict_get)
        d = {'known': 1}
        get_ret = dict_get(d)
        get_op_ret = dict_get_op(d)
        print(get_ret)
        print(get_op_ret)
        self.assertSequenceEqual([1, 1, None, 10000, 1], get_ret[:-1])
        self.assertSequenceEqual([1, 2, 3], get_ret[-1])
        self.assertSequenceEqual([1, 1, None, 10000, 1], get_op_ret[:-1])
        self.assertSequenceEqual([1, 2, 3], get_op_ret[-1])

    def test_general_dict(self):
        def general_get(a: Any) -> List:
            r = []
            r.append(a.get('known'))
            r.append(a.get('known', 10000))
            r.append(a.get('unknown'))
            r.append(a.get('unknown', 10000))
            r.append(a.get('known', [1, 2, 3]))
            r.append(a.get('unknown', [1, 2, 3]))
            return r

        general_get_op = matx.script(general_get)
        d = {'known': 1}
        get_ret = general_get(d)
        get_op_ret = general_get_op(d)
        print(get_ret)
        print(get_op_ret)
        self.assertSequenceEqual([1, 1, None, 10000, 1], get_ret[:-1])
        self.assertSequenceEqual([1, 2, 3], get_ret[-1])
        self.assertSequenceEqual([1, 1, None, 10000, 1], get_op_ret[:-1])
        self.assertSequenceEqual([1, 2, 3], get_op_ret[-1])

    def test_dict_pop(self):
        def dict_pop() -> Tuple[Dict, List]:
            a = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
            ret = []
            ret.append(a.pop('a'))
            ret.append(a.pop('a', None))
            ret.append(a.pop('b', 3))
            ret.append(a.pop('e', None))
            ret.append(a.pop('e', 4))
            return a, ret

        dict_pop_op = matx.script(dict_pop)
        x, y = dict_pop()
        self.assertEqual(set({'c', 'd'}), set(x.keys()))
        self.assertListEqual([1, None, 2, None, 4], list(y))

        x, y = dict_pop_op()
        self.assertEqual(set({'c', 'd'}), set(x.keys()))
        self.assertListEqual([1, None, 2, None, 4], list(y))

        def dict_exception_pop() -> None:
            a = {'a': 1}
            a.pop('b')
        dict_exception_pop_op = matx.script(dict_exception_pop)
        self.assertTrue(catch_raise(dict_exception_pop))
        self.assertTrue(catch_raise(dict_exception_pop_op))

    def test_generic_pop(self):
        def generic_dict_pop() -> Tuple[Dict, List]:
            a = [{'a': 1, 'b': 2, 'c': 3, 'd': 4}][0]
            ret = []
            ret.append(a.pop('a'))
            ret.append(a.pop('a', None))
            ret.append(a.pop('b', 3))
            ret.append(a.pop('e', None))
            ret.append(a.pop('e', 4))
            return a, ret

        generic_dict_pop_op = matx.script(generic_dict_pop)
        x, y = generic_dict_pop()
        self.assertEqual(set({'c', 'd'}), set(x.keys()))
        self.assertListEqual([1, None, 2, None, 4], list(y))

        x, y = generic_dict_pop_op()
        self.assertEqual(set({'c', 'd'}), set(x.keys()))
        self.assertListEqual([1, None, 2, None, 4], list(y))

        def generic_dict_exception_pop() -> None:
            a = [{'a': 1}][0]
            a.pop('b')
        generic_dict_exception_pop_op = matx.script(generic_dict_exception_pop)
        self.assertTrue(catch_raise(generic_dict_exception_pop))
        self.assertTrue(catch_raise(generic_dict_exception_pop_op))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
