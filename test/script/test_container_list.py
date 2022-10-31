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

from typing import List
from typing import Any
import unittest
import random
import matx


class TestContainerList(unittest.TestCase):
    def test_list_append(self):
        def list_append(container: List, item: Any) -> List:
            container.append(item)
            return container

        list_append_op = matx.script(list_append)
        self.assertEqual(list_append(["1"], 2), ["1", 2])
        self.assertEqual(list_append_op(["1"], 2), ["1", 2])
        self.assertEqual(list_append_op(list(["1"]), 2), ["1", 2])

    def test_generic_append(self):
        def generic_append(container: Any, item: Any) -> Any:
            container.append(item)
            return container

        generic_append_op = matx.script(generic_append)
        self.assertEqual(generic_append(["1"], 2), ["1", 2])
        self.assertEqual(generic_append_op(["1"], 2), ["1", 2])
        self.assertEqual(generic_append_op(list(["1"]), 2), ["1", 2])

    def test_list_repeat(self):
        def list_repeat(times: int) -> int:
            origin = [1, 2, 4]
            b = origin * times
            return len(origin) + len(b) + b[5]

        def list_repeat_cons(times: int) -> int:
            origin = list([1, 2, 4])
            b = origin * times
            return len(origin) + len(b) + b[5]

        list_repeat_op = matx.script(list_repeat)
        list_repeat_cons_op = matx.script(list_repeat_cons)
        self.assertEqual(list_repeat(3), 16)
        self.assertEqual(list_repeat_op(3), 16)
        self.assertEqual(list_repeat_cons_op(3), 16)

        def list_repeat_object_type_times() -> int:
            origin = [1, 2, 4]
            b = origin * origin[1]
            return len(origin) + len(b) + b[5]

        def list_repeat_object_type_times_cons() -> int:
            origin = list([1, 2, 4])
            b = origin * origin[1]
            return len(origin) + len(b) + b[5]

        list_repeat_object_type_times_op = matx.script(list_repeat_object_type_times)
        list_repeat_object_type_times_cons_op = matx.script(list_repeat_object_type_times_cons)
        self.assertEqual(list_repeat_object_type_times(), 13)
        self.assertEqual(list_repeat_object_type_times_op(), 13)
        self.assertEqual(list_repeat_object_type_times_cons_op(), 13)

    def test_generic_repeat(self):
        def generic_repeat(times: int) -> int:
            origin = [[1, 2, 4]]
            origin = origin[0]
            b = origin * times
            return len(origin) + len(b) + b[5]

        def generic_repeat_cons(times: int) -> int:
            origin = list([list([1, 2, 4])])
            origin = origin[0]
            b = origin * times
            return len(origin) + len(b) + b[5]

        generic_repeat_op = matx.script(generic_repeat)
        generic_repeat_cons_op = matx.script(generic_repeat_cons)
        self.assertEqual(generic_repeat(3), 16)
        self.assertEqual(generic_repeat_op(3), 16)
        self.assertEqual(generic_repeat_cons_op(3), 16)

    def test_list_extend(self):

        def list_extend() -> int:
            a = [1, 2, 'a']
            b = [5, 6]
            a.extend(b)
            return a[4]

        def list_extend_cons() -> int:
            a = list([1, 2, 'a'])
            b = list([5, 6])
            a.extend(b)
            return a[4]

        list_extend_op = matx.script(list_extend)
        list_extend_op_cons = matx.script(list_extend_cons)
        self.assertEqual(list_extend(), 6)
        self.assertEqual(list_extend_op(), 6)
        self.assertEqual(list_extend_op_cons(), 6)

        def list_extend2() -> List:
            a = [5]
            b = [a, a]
            l = []
            l.extend(b[0])
            l.extend(b[1])
            return l

        def list_extend2_cons() -> List:
            a = list([5])
            b = list([a, a])
            l = list()
            l.extend(b[0])
            l.extend(b[1])
            return l

        list_extend2_op = matx.script(list_extend2)
        list_extend2_cons_op = matx.script(list_extend2_cons)
        self.assertEqual(list_extend2(), [5, 5])
        self.assertEqual(list_extend2_op(), [5, 5])
        self.assertEqual(list_extend2_cons_op(), [5, 5])

    def test_generic_extend(self):

        def generic_extend(a: Any) -> int:
            b = [5, 6]
            a.extend(b)
            return a[4]

        generic_extend_op = matx.script(generic_extend)
        self.assertEqual(generic_extend([1, 2, 'a']), 6)
        self.assertEqual(generic_extend_op([1, 2, 'a']), 6)

    def test_list_in(self):
        def list_in(x: List) -> bool:
            if 2 in x:
                return True
            else:
                return False

        list_in_op = matx.script(list_in)
        self.assertEqual(list_in([1, 3, 5]), False)
        self.assertEqual(list_in([1, 2, 3]), True)
        self.assertEqual(list_in_op([1, 3, 5]), False)
        self.assertEqual(list_in_op([1, 2, 3]), True)

    def test_general_in(self):
        def general_in(x: Any) -> bool:
            if 2 in x:
                return True
            else:
                return False

        general_in_op = matx.script(general_in)
        self.assertEqual(general_in(list([1, 3, 5])), False)
        self.assertEqual(general_in(list([1, 2, 3])), True)
        self.assertEqual(general_in_op(list([1, 3, 5])), False)
        self.assertEqual(general_in_op(list([1, 2, 3])), True)

    def test_list_notin(self):
        def list_notin(x: List) -> bool:
            if 2 not in x:
                return True
            else:
                return False

        list_notin_op = matx.script(list_notin)
        self.assertEqual(list_notin(list([1, 3, 5])), True)
        self.assertEqual(list_notin(list([1, 2, 3])), False)
        self.assertEqual(list_notin_op(list([1, 3, 5])), True)
        self.assertEqual(list_notin_op(list([1, 2, 3])), False)

    def test_general_notin(self):
        def general_notin(x: Any) -> bool:
            if 2 not in x:
                return True
            else:
                return False

        general_notin_op = matx.script(general_notin)
        self.assertEqual(general_notin(list([1, 3, 5])), True)
        self.assertEqual(general_notin(list([1, 2, 3])), False)
        self.assertEqual(general_notin_op(list([1, 3, 5])), True)
        self.assertEqual(general_notin_op(list([1, 2, 3])), False)

    def test_list_subscript(self):

        def list_subscript(a: list) -> list:
            return a[1:3]

        list_subscript_op = matx.script(list_subscript)
        self.assertEqual(list_subscript(list([1, 2, 3, 4])), list([2, 3]))
        self.assertEqual(list_subscript_op(list([1, 2, 3, 4])), list([2, 3]))

    def test_general_subscript(self):

        def general_subscript(a: Any) -> list:
            return a[1:3]

        general_subscript_op = matx.script(general_subscript)
        self.assertEqual(general_subscript(list([1, 2, 3, 4])), list([2, 3]))
        self.assertEqual(general_subscript_op(list([1, 2, 3, 4])), list([2, 3]))

    # TODO: add comment stmt
    # def test_list_reserve(self):
    #     def list_reserve(a: list, new_size: int) -> list:
    #         a.reserve(new_size)
    #         return a

    #     a = list()
    #     list_reserve(a, 100)
    #     self.assertEqual(a.capacity(), 100)
    #     list_reserve(a, -1)
    #     self.assertEqual(a.capacity(), 100)
    #     list_reserve(a, 50)
    #     self.assertEqual(a.capacity(), 100)

    #     list_reserve_op = matx.script(list_reserve)
    #     b = list()
    #     list_reserve_op(b, 100)
    #     self.assertEqual(b.capacity(), 100)
    #     list_reserve_op(b, -1)
    #     self.assertEqual(b.capacity(), 100)
    #     list_reserve_op(b, 50)
    #     self.assertEqual(b.capacity(), 100)

    # def test_general_reserve(self):

    #     def general_reserve(a: Any, new_size: int) -> Any:
    #         a.reserve(new_size)
    #         return a

    #     a = list()
    #     general_reserve(a, 100)
    #     self.assertEqual(a.capacity(), 100)
    #     general_reserve(a, -1)
    #     self.assertEqual(a.capacity(), 100)
    #     general_reserve(a, 50)
    #     self.assertEqual(a.capacity(), 100)

    #     general_reserve_op = matx.script(general_reserve)
    #     b = list()
    #     general_reserve_op(b, 100)
    #     self.assertEqual(b.capacity(), 100)
    #     general_reserve_op(b, -1)
    #     self.assertEqual(b.capacity(), 100)
    #     general_reserve_op(b, 50)
    #     self.assertEqual(b.capacity(), 100)

    def test_list_iterator(self):
        def list_iterator(a: list) -> list:
            result = list()
            for x in a:
                result.append(x)
            return result

        list_iterator_op = matx.script(list_iterator)
        data = list([1, 3, 5])
        self.assertEqual(data, list_iterator(data))
        self.assertEqual(data, list_iterator_op(data))

    def test_generic_iterator(self):
        def generic_iterator(a: Any) -> Any:
            result = list()
            for x in a:
                result.append(x)
            return result

        generic_iterator_op = matx.script(generic_iterator)
        data = list([1, 3, 5])
        self.assertEqual(data, generic_iterator(data))
        self.assertEqual(data, generic_iterator_op(data))

    def test_add(self):
        def list_add(a: List, b: List) -> List:
            return a + b

        list_add_op = matx.script(list_add)
        a = list([1, 2, 3])
        b = list([4, 5, 6])
        c = list_add(a, b)
        c_op = list_add_op(a, b)
        self.assertSequenceEqual(c, list(range(1, 7)))
        self.assertSequenceEqual(c_op, list(range(1, 7)))

    def test_iadd(self):
        def list_iadd(a: List, b: List) -> List:
            a += b
            return a

        list_iadd_op = matx.script(list_iadd)
        a = list([1, 2, 3])
        b = list([4, 5, 6])
        c = list([7, 8, 9])
        d = list_iadd(a, b)
        self.assertSequenceEqual(a, list(range(1, 7)))
        self.assertSequenceEqual(d, list(range(1, 7)))
        d = list_iadd_op(a, c)  # "+=" bug, issue 131
        self.assertSequenceEqual(d, list(range(1, 10)))

    def test_list_remove(self):
        def list_remove(l: List, item: Any) -> List:
            l.remove(item)
            return l

        list_remove_op = matx.script(list_remove)
        l = list([1, 2, 3, 1])
        r = list_remove_op(l, 1)
        self.assertEqual(r, list([2, 3, 1]))
        l2 = list([1, 2, 3, 1])
        l2.remove(1)
        self.assertEqual(r, l2)

    def test_general_remove(self):
        def general_remove(l: Any, item: Any) -> Any:
            l.remove(item)
            return l

        general_remove_op = matx.script(general_remove)
        l = list([1, 2, 3, 1])
        l = general_remove_op(l, 1)
        self.assertEqual(l, list([2, 3, 1]))
        l2 = list([1, 2, 3, 1])
        l2.remove(1)
        self.assertEqual(l, l2)

    def test_list_insert(self):
        def list_insert(l: List, index: int, item: Any) -> List:
            l.insert(index, item)
            return l

        list_insert_op = matx.script(list_insert)
        l = list([1, 2, 3, 4])
        r = list_insert_op(l, 100, 5)
        self.assertEqual(r, list([1, 2, 3, 4, 5]))
        l2 = list([1, 2, 3, 4])
        l2.insert(100, 5)
        self.assertEqual(r, l2)

        l = list([1, 2, 3, 4])
        r = list_insert_op(l, -100, 5)
        self.assertEqual(r, list([5, 1, 2, 3, 4]))
        l2 = list([1, 2, 3, 4])
        l2.insert(-100, 5)
        self.assertEqual(r, l2)

    def test_general_insert(self):
        def general_insert(l: Any, index: int, item: Any) -> Any:
            l.insert(index, item)
            return l

        general_insert_op = matx.script(general_insert)
        l = list([1, 2, 3, 4])
        r = general_insert_op(l, 100, 5)
        self.assertEqual(r, list([1, 2, 3, 4, 5]))
        l2 = list([1, 2, 3, 4])
        l2.insert(100, 5)
        self.assertEqual(r, l2)

        l = list([1, 2, 3, 4])
        r = general_insert_op(l, -100, 5)
        self.assertEqual(r, list([5, 1, 2, 3, 4]))
        l2 = list([1, 2, 3, 4])
        l2.insert(-100, 5)
        self.assertEqual(r, l2)

    def test_general_index(self):

        def list_index_int(l: Any, target: Any) -> Any:
            return l.index(target)

        def list_index_with_start(l: Any, target: Any, start: int) -> Any:
            return l.index(target, start)

        def list_index_with_start_end(l: Any, target: Any, start: int, end: int) -> Any:
            return l.index(target, start, end)

        list_index_int_op = matx.script(list_index_int)
        list_index_with_start_op = matx.script(list_index_with_start)
        list_index_with_start_end_op = matx.script(list_index_with_start_end)

        l_int = list([1, 2, 3, 4, 5])
        idx = list_index_int_op(l_int, 3)
        self.assertEqual(idx, 2)
        idx = list_index_int_op(l_int, 4)
        self.assertEqual(idx, 3)

        idx = list_index_with_start_op(l_int, 3, 0)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 1)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 2)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, -3)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 3)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 100)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, -1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        idx = list_index_with_start_end_op(l_int, 3, 2, 3)
        self.assertEqual(idx, 2)

        idx = list_index_with_start_end_op(l_int, 3, -3, -1)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op(l_int, 3, 2, 2)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end(l_int, 3, 4, 1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

    def test_list_index(self):

        def list_index_int(l: List[int], target: int) -> int:
            i: int = l.index(target)
            return i

        def list_index_with_start(l: List[int], target: int, start: int) -> int:
            return l.index(target, start)

        def list_index_with_start_end(l: List[int], target: int, start: int, end: int) -> int:
            return l.index(target, start, end)

        def list_index_with_start_end2(target: int, start: int, end: int) -> int:
            l: List[int] = list([1, 2, 3, 4, 5])
            return l.index(target, start, end)

        list_index_int_op = matx.script(list_index_int)
        list_index_with_start_op = matx.script(list_index_with_start)
        list_index_with_start_end_op = matx.script(list_index_with_start_end)
        list_index_with_start_end_op2 = matx.script(list_index_with_start_end2)

        l_int: List = list([1, 2, 3, 4, 5])
        idx = list_index_int_op(l_int, 3)
        self.assertEqual(idx, 2)
        idx = list_index_int_op(l_int, 4)
        self.assertEqual(idx, 3)

        idx = list_index_with_start_op(l_int, 3, 0)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 1)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 2)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, -3)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 3)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 100)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, -1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        idx = list_index_with_start_end_op(l_int, 3, 2, 3)
        self.assertEqual(idx, 2)

        idx = list_index_with_start_end_op(l_int, 3, -3, -1)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op(l_int, 3, 2, 2)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end(l_int, 3, 4, 1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        idx = list_index_with_start_end_op2(3, 2, 3)
        self.assertEqual(idx, 2)

        idx = list_index_with_start_end_op2(3, -3, -1)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op2(3, 2, 2)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op2(3, 4, 1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

    def test_list_pop(self):
        def list_pop(l: List, index: int) -> Any:
            if index == -1:
                return l.pop(), l
            else:
                return l.pop(index), l

        list_pop_op = matx.script(list_pop)
        l = list([1, 2, 3, 4, 5])
        a, l = list_pop_op(l, -1)
        self.assertEqual(a, 5)
        self.assertEqual(l, [1, 2, 3, 4])
        b, l = list_pop_op(l, 0)
        self.assertEqual(b, 1)
        self.assertEqual([2, 3, 4], l)
        l2 = list([1, 2, 3, 4, 5])
        c = l2.pop(-1)
        self.assertEqual(c, 5)
        self.assertEqual(l2, [1, 2, 3, 4])
        d = l2.pop(-2)
        self.assertEqual(d, 3)
        self.assertEqual(l2, [1, 2, 4])

    def test_general_pop(self):
        def general_pop(l: Any, index: int) -> Any:
            if index == -1:
                return l.pop(), l
            else:
                return l.pop(index), l

        general_pop_op = matx.script(general_pop)
        l = list([1, 2, 3, 4, 5])
        a, l = general_pop_op(l, -1)
        self.assertEqual(a, 5)
        self.assertEqual(l, [1, 2, 3, 4])
        b, l = general_pop_op(l, 0)
        self.assertEqual(b, 1)
        print(l)
        self.assertEqual([2, 3, 4], l)

    def test_list_slice(self):
        def list_slice(a: List[int]) -> List[int]:
            slice_size = len(a) + 3
            return a[:slice_size]

        python_list = [1, 2, 3, 4, 5, 6]
        matx_list = list(python_list)
        python_ret = python_list[:9]
        matx_ret = matx_list[:9]
        matx_script_slice = matx.script(list_slice)
        matx_script_ret = matx_script_slice(python_list)

        self.assertEqual(python_list, python_ret)
        self.assertEqual(python_list, matx_ret)
        self.assertEqual(python_list, matx_script_ret)

    def test_list_clear(self):
        def list_clear(l: List) -> Any:
            l.clear()
            return l

        l1 = list([1, 2, 3])
        l1 = list_clear(l1)
        self.assertEqual(len(l1), 0)

        matx_script_clear = matx.script(list_clear)
        l2 = list([1, 2, 3])
        l2 = matx_script_clear(l2)
        self.assertEqual(len(l2), 0)

    def test_list_basic_sort(self):
        def list_basic_sort(a: List) -> Any:
            a.sort()
            return a

        def any_basic_sort(a: Any) -> Any:
            a.sort()
            return a

        data = [random.randint(0, 100) for i in range(10)]
        py_ret = list_basic_sort(data)
        tx_list_basic_sort = matx.script(list_basic_sort)
        tx_any_basic_sort = matx.script(any_basic_sort)
        tx_ret1 = tx_list_basic_sort(data)
        tx_ret2 = tx_any_basic_sort(data)
        self.assertSequenceEqual(py_ret, tx_ret1)
        self.assertSequenceEqual(py_ret, tx_ret2)

    def test_list_reverse_sort(self):
        def list_reverse_sort(a: List) -> Any:
            a.sort(reverse=True)
            return a

        def any_reverse_sort(a: Any) -> Any:
            a.sort(reverse=True)
            return a

        data = [random.randint(0, 100) for i in range(10)]
        py_ret = list_reverse_sort(data)
        tx_list_reverse_sort = matx.script(list_reverse_sort)
        tx_any_reverse_sort = matx.script(any_reverse_sort)
        tx_ret1 = tx_list_reverse_sort(data)
        tx_ret2 = tx_any_reverse_sort(data)
        self.assertSequenceEqual(py_ret, tx_ret1)
        self.assertSequenceEqual(py_ret, tx_ret2)

    def test_list_key_sort(self):
        def my_key(x: int) -> int:
            return -x

        def list_key_sort(a: List) -> Any:
            a.sort(key=my_key)
            return a

        def any_key_sort(a: Any) -> Any:
            a.sort(key=my_key)
            return a

        data = [random.randint(0, 100) for i in range(10)]
        py_ret = list_key_sort(data)
        tx_list_key_sort = matx.script(list_key_sort)
        tx_any_key_sort = matx.script(any_key_sort)
        tx_ret1 = tx_list_key_sort(data)
        tx_ret2 = tx_any_key_sort(data)
        self.assertSequenceEqual(py_ret, tx_ret1)
        self.assertSequenceEqual(py_ret, tx_ret2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
