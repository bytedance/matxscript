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

import random
import unittest
import math

import matx
from typing import Any
from typing import List


def _test_generator(n, func, args):
    import time
    print(n, 'times', func.__name__)
    total = 0.0
    sqsum = 0.0
    smallest = 1e10
    largest = -1e10
    t0 = time.perf_counter()
    for i in range(n):
        x = func(*args)
        total += x
        sqsum = sqsum + x * x
        smallest = min(x, smallest)
        largest = max(x, largest)
    t1 = time.perf_counter()
    print(round(t1 - t0, 3), 'sec,', end=' ')
    avg = total / n
    stddev = math.sqrt(sqsum / n - avg * avg)
    print('avg %g, stddev %g, min %g, max %g\n' %
          (avg, stddev, smallest, largest))


class TestRandom(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_random(self):
        @matx.script
        def random_random() -> float:
            return random.random()

        print(random_random())

    def test_utils(self):
        N = 2000

        @matx.script
        def random_random() -> float:
            return random.random()

        _test_generator(N, random_random, ())

        @matx.script
        def random_normalvariate(mu: float, sigma: float) -> float:
            return random.normalvariate(mu, sigma)

        _test_generator(N, random_normalvariate, (0.0, 1.0))

        @matx.script
        def random_lognormvariate(mu: float, sigma: float) -> float:
            return random.lognormvariate(mu, sigma)

        _test_generator(N, random_lognormvariate, (0.0, 1.0))

        @matx.script
        def random_vonmisesvariate(mu: float, kappa: float) -> float:
            return random.vonmisesvariate(mu, kappa)

        _test_generator(N, random_vonmisesvariate, (0.0, 1.0))

        @matx.script
        def random_gammavariate(alpha: float, beta: float) -> float:
            return random.gammavariate(alpha, beta)

        _test_generator(N, random_gammavariate, (0.01, 1.0))
        _test_generator(N, random_gammavariate, (0.1, 1.0))
        _test_generator(N, random_gammavariate, (0.1, 2.0))
        _test_generator(N, random_gammavariate, (0.5, 1.0))
        _test_generator(N, random_gammavariate, (0.9, 1.0))
        _test_generator(N, random_gammavariate, (1.0, 1.0))
        _test_generator(N, random_gammavariate, (2.0, 1.0))
        _test_generator(N, random_gammavariate, (20.0, 1.0))
        _test_generator(N, random_gammavariate, (200.0, 1.0))

        @matx.script
        def random_gauss(mu: float, sigma: float) -> float:
            return random.gauss(mu, sigma)

        _test_generator(N, random_gauss, (0.0, 1.0))

        @matx.script
        def random_betavariate(alpha: float, beta: float) -> float:
            return random.betavariate(alpha, beta)

        _test_generator(N, random_betavariate, (3.0, 3.0))

        @matx.script
        def random_triangular(low: float = 0.0, high: float = 1.0, mode: Any = None) -> float:
            return random.triangular(low, high, mode)

        _test_generator(N, random_triangular, (0.0, 1.0, 1.0 / 3.0))

    def test_diff(self):
        def random_int() -> List[float]:
            random.seed(10)
            return [random.random() for _ in range(10)]

        a = random_int()
        b = matx.script(random_int)()
        b = [f for f in b]
        print(a)
        print(b)
        self.assertEqual(a, b)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
