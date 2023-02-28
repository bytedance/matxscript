import unittest
from matx.ir.kernelgen.kernel_typing import broadcast
import sympy


class TestBroadCast(unittest.TestCase):

    def test_basic_numeric_eq_shape(self):
        shape1 = [2, 4]
        shape2 = [2, 4]
        expected_result_shape = [2, 4]
        expected_broadcast_shape1 = [2, 4]
        expected_broadcast_shape2 = [2, 4]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_basic_numeric_var_shape1(self):
        shape1 = [2, 4, 5]
        shape2 = [4, 5]
        expected_result_shape = [2, 4, 5]
        expected_broadcast_shape1 = [2, 4, 5]
        expected_broadcast_shape2 = [None, 4, 5]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_basic_numeric_var_shape2(self):
        shape1 = [5]
        shape2 = [3, 2, 2, 4, 5]
        expected_result_shape = [3, 2, 2, 4, 5]
        expected_broadcast_shape1 = [None, None, None, None, 5]
        expected_broadcast_shape2 = [3, 2, 2, 4, 5]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_basic_numeric_var_shape3(self):
        shape1 = [1]
        shape2 = [3, 2, 2, 4, 5]
        expected_result_shape = [3, 2, 2, 4, 5]
        expected_broadcast_shape1 = [1]
        expected_broadcast_shape2 = [3, 2, 2, 4, 5]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_eq_shape(self):
        N = sympy.Symbol('N')
        M = sympy.Symbol('M')
        shape1 = [N, M]
        shape2 = [N, M]
        expected_result_shape = [N, M]
        expected_broadcast_shape1 = [N, M]
        expected_broadcast_shape2 = [N, M]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_var_shape(self):
        N = sympy.Symbol('N')
        M = sympy.Symbol('M')
        shape1 = [N, M]
        shape2 = [M]
        expected_result_shape = [N, M]
        expected_broadcast_shape1 = [N, M]
        expected_broadcast_shape2 = [None, M]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_expression_shape(self):
        N = sympy.Symbol('N')
        M = sympy.Symbol('M')
        shape1 = [N + N, M + N]
        shape2 = [N + N, N + M]
        expected_result_shape = [N + N, M + N]
        expected_broadcast_shape1 = [N + N, M + N]
        expected_broadcast_shape2 = [N + N, M + N]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_with_scalar_shape(self):
        N = sympy.Symbol('N')
        M = sympy.Symbol('M')
        shape1 = [1]
        shape2 = [N, M]
        expected_result_shape = [N, M]
        expected_broadcast_shape1 = [1]
        expected_broadcast_shape2 = [N, M]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_with_numeric_shape(self):
        M = sympy.Symbol('M')
        shape1 = [10, M]
        shape2 = [M]
        expected_result_shape = [10, M]
        expected_broadcast_shape1 = [10, M]
        expected_broadcast_shape2 = [None, M]

        rc = broadcast(shape1, shape2)
        actual_result_shape = rc[0]
        actual_broadcast_shape1 = rc[1]
        actual_broadcast_shape2 = rc[2]

        self.assertEqual(expected_result_shape, actual_result_shape)
        self.assertEqual(expected_broadcast_shape1, actual_broadcast_shape1)
        self.assertEqual(expected_broadcast_shape2, actual_broadcast_shape2)

    def test_symbol_with_value_shape_exception(self):
        N = sympy.Symbol('N')
        M = sympy.Symbol('M')
        shape1 = [2, 3]
        shape2 = [N, M]

        self.assertRaises(SyntaxError, broadcast, shape1, shape2)
        self.assertRaises(SyntaxError, broadcast, shape2, shape1)
