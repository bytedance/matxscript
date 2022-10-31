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
from typing import List, Set, Tuple
from typing import Any
import unittest
import matx


class TestContainerSet(unittest.TestCase):

    def test_builtin_set(self):
        @matx.script
        def set_add(container: Set, item: Any) -> Set:
            container.add(item)
            # add one more time to test
            container.add(item)
            return container

        @matx.script
        def generic_add(container: Any, item: Any) -> Any:
            container.add(item)
            container.add(item)
            return container

        test_data = {"init"}
        self.assertEqual(set_add(test_data, 10), matx.Set(["init", 10]))
        self.assertEqual(generic_add(test_data, 10), matx.Set(["init", 10]))

    def test_set_in(self):

        @matx.script
        def set_in(x: Set) -> bool:
            if 2 in x:
                return True
            else:
                return False

        @matx.script
        def generic_in(x: Any) -> bool:
            if 2 in x:
                return True
            else:
                return False

        self.assertEqual(set_in({1, 3, 4}), False)
        self.assertEqual(set_in({1, 2, 3}), True)
        self.assertEqual(generic_in({1, 3, 4}), False)
        self.assertEqual(generic_in({1, 2, 3}), True)

    def test_set_clear(self):

        @matx.script
        def set_clear(x: Set) -> int:
            x.clear()
            return len(x)

        @matx.script
        def generic_clear(x: Any) -> int:
            x.clear()
            return len(x)

        self.assertEqual(set_clear({1, 2, 3}), 0)
        self.assertEqual(generic_clear(matx.Set([1, 2, 3])), 0)

    def test_set_reserve(self):

        @matx.script
        def set_reserve(x: matx.Set, new_size: int) -> matx.Set:
            x.reserve(new_size)
            return x

        x = matx.Set()
        set_reserve(x, 100)
        self.assertGreaterEqual(x.bucket_count(), 100)
        set_reserve(x, 50)
        self.assertGreaterEqual(x.bucket_count(), 50)
        set_reserve(x, -1)
        self.assertGreaterEqual(x.bucket_count(), 50)

    def test_general_reserve(self):

        @matx.script
        def general_reserve(x: Any, new_size: int) -> Any:
            x.reserve(new_size)
            return x

        x = matx.Set()
        general_reserve(x, 100)
        self.assertGreaterEqual(x.bucket_count(), 100)
        general_reserve(x, 50)
        self.assertGreaterEqual(x.bucket_count(), 50)
        general_reserve(x, -1)
        self.assertGreaterEqual(x.bucket_count(), 50)

    def test_set_iter(self):
        def set_iter(s: Set) -> List:
            ret = []
            for item in s:
                ret.append(item)
            return ret
        l = [1, 2, 3, 4]
        s = set(l)
        op = matx.script(set_iter)
        res = sorted(set_iter(s))
        op_res = sorted(op(s))
        self.assertSequenceEqual(l, res)
        self.assertSequenceEqual(l, op_res)

    def test_set_general_iter(self):
        def set_general_iter(s: Any) -> List:
            ret = []
            for item in s:
                ret.append(item)
            return ret
        l = [1, 2, 3, 4]
        s = set(l)
        op = matx.script(set_general_iter)
        res = sorted(set_general_iter(s))
        op_res = sorted(op(s))
        self.assertSequenceEqual(l, res)
        self.assertSequenceEqual(l, op_res)

    def test_set_difference(self):
        def set_difference(s: Set) -> List:
            ret = []
            ret.append(s.difference({1, 2}))
            ret.append(s.difference([1, 2]))
            ret.append(s.difference([1, 2], {4}))
            ret.append(s.difference([4]))
            ret.append(s.difference())
            return ret

        set_difference_op = matx.script(set_difference)
        s = {1, 2, 3}
        ret = set_difference(s)
        self.assertEqual(ret[0], {3})
        self.assertEqual(ret[1], {3})
        self.assertEqual(ret[2], {3})
        self.assertEqual(ret[3], {1, 2, 3})
        self.assertEqual(ret[4], {1, 2, 3})

        op_ret = set_difference_op(s)
        self.assertEqual(op_ret[0], {3})
        self.assertEqual(op_ret[1], {3})
        self.assertEqual(op_ret[2], {3})
        self.assertEqual(op_ret[3], {1, 2, 3})
        self.assertEqual(op_ret[4], {1, 2, 3})

    def test_general_difference(self):
        def general_difference(s: Any) -> List:
            ret = []
            ret.append(s.difference({1, 2}))
            ret.append(s.difference([1, 2]))
            ret.append(s.difference([1, 2], {4}))
            ret.append(s.difference([4]))
            ret.append(s.difference())
            return ret

        general_difference_op = matx.script(general_difference)
        s = {1, 2, 3}
        ret = general_difference(s)
        self.assertEqual(ret[0], {3})
        self.assertEqual(ret[1], {3})
        self.assertEqual(ret[2], {3})
        self.assertEqual(ret[3], {1, 2, 3})
        self.assertEqual(ret[4], {1, 2, 3})

        op_ret = general_difference_op(s)
        self.assertEqual(op_ret[0], {3})
        self.assertEqual(op_ret[1], {3})
        self.assertEqual(op_ret[2], {3})
        self.assertEqual(op_ret[3], {1, 2, 3})
        self.assertEqual(op_ret[4], {1, 2, 3})

    def test_set_discard(self):
        def set_discard(s: Set, a: Any) -> Set:
            s.discard(a)
            return s
        set_discard_op = matx.script(set_discard)
        s = {1, 2, 3, 4, 5}
        self.assertEqual(set_discard(s, 1), {2, 3, 4, 5})
        self.assertEqual(set_discard(s, 6), {2, 3, 4, 5})
        self.assertEqual(set_discard_op(s, 1), {2, 3, 4, 5})
        self.assertEqual(set_discard_op(s, 2), {3, 4, 5})

    def test_general_discard(self):
        def general_discard(s: Any, a: Any) -> Set:
            s.discard(a)
            return s
        general_discard_op = matx.script(general_discard)
        s = {1, 2, 3, 4, 5}
        self.assertEqual(general_discard(s, 1), {2, 3, 4, 5})
        self.assertEqual(general_discard(s, 6), {2, 3, 4, 5})
        self.assertEqual(general_discard_op(s, 1), {2, 3, 4, 5})
        self.assertEqual(general_discard_op(s, 2), {3, 4, 5})

    def test_set_difference_update(self):
        def set_difference_update() -> List:
            ret = []
            s = {1, 2, 3}
            s.difference_update({2, 3}, [4])
            ret.append(s)

            s = {1, 2, 3}
            s.difference_update()
            ret.append(s)

            s = {1, 2, 3}
            s.difference_update({4})
            ret.append(s)
            return ret

        set_difference_update_op = matx.script(set_difference_update)
        ret = set_difference_update()
        self.assertEqual(ret[0], {1})
        self.assertEqual(ret[1], {1, 2, 3})
        self.assertEqual(ret[2], {1, 2, 3})

        op_ret = set_difference_update_op()
        self.assertEqual(op_ret[0], {1})
        self.assertEqual(op_ret[1], {1, 2, 3})
        self.assertEqual(op_ret[2], {1, 2, 3})

    def test_general_difference_update(self):
        def general_difference_update() -> List:
            ret = []
            s = [{1, 2, 3}][0]
            s.difference_update({2, 3}, [4])
            ret.append(s)

            s = [{1, 2, 3}][0]
            s.difference_update()
            ret.append(s)

            s = [{1, 2, 3}][0]
            s.difference_update({4})
            ret.append(s)
            return ret

        general_difference_update_op = matx.script(general_difference_update)
        ret = general_difference_update()
        self.assertEqual(ret[0], {1})
        self.assertEqual(ret[1], {1, 2, 3})
        self.assertEqual(ret[2], {1, 2, 3})

        op_ret = general_difference_update_op()
        self.assertEqual(op_ret[0], {1})
        self.assertEqual(op_ret[1], {1, 2, 3})
        self.assertEqual(op_ret[2], {1, 2, 3})

    def test_set_update(self):
        def set_update() -> Any:
            a = {1, 2, 3}
            a.update([1, 4], {2, 5}, {1, 3})
            return a

        set_update_op = matx.script(set_update)
        ret = set_update()
        self.assertEqual({1, 2, 3, 4, 5}, set(ret))
        op_ret = set_update_op()
        self.assertEqual({1, 2, 3, 4, 5}, set(op_ret))

    def test_generic_update(self):
        def generic_set_update() -> Any:
            a = [{1, 2, 3}][0]
            a.update([1, 4], {2, 5}, {1, 3})
            return a

        generic_set_update_op = matx.script(generic_set_update)
        ret = generic_set_update()
        self.assertEqual({1, 2, 3, 4, 5}, set(ret))
        op_ret = generic_set_update_op()
        self.assertEqual({1, 2, 3, 4, 5}, set(op_ret))

    def test_set_union(self):
        def set_union() -> Tuple[Any, Any]:
            a = {1, 2, 3}
            b = a.union([1, 4], {2, 5}, {1, 3})
            return a, b

        set_union_op = matx.script(set_union)
        a, b = set_union()
        self.assertEqual({1, 2, 3}, set(a))
        self.assertEqual({1, 2, 3, 4, 5}, set(b))

        a, b = set_union_op()
        self.assertEqual({1, 2, 3}, set(a))
        self.assertEqual({1, 2, 3, 4, 5}, set(b))

    def test_generic_union(self):
        def generic_set_union() -> Tuple[Any, Any]:
            a = [{1, 2, 3}][0]
            b = a.union([1, 4], {2, 5}, {1, 3})
            return a, b

        generic_set_union_op = matx.script(generic_set_union)
        a, b = generic_set_union()
        self.assertEqual({1, 2, 3}, set(a))
        self.assertEqual({1, 2, 3, 4, 5}, set(b))

        a, b = generic_set_union_op()
        self.assertEqual({1, 2, 3}, set(a))
        self.assertEqual({1, 2, 3, 4, 5}, set(b))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
