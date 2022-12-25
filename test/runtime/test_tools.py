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
import matx
from matx.tools import LGBMPredictor

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestLGBM(unittest.TestCase):
    def test_eval(self):
        feature = [2, -0.2226236411773681, -0.8203237354755402, -8.7032166,
                   -8.387475499999999, 4, 0, 0.9999975000062501, -0.5301455855369568,
                   -6.073882102966309, -4.049746978282928, 69.62790210378144,
                   -19.428411155939102, -20.248734891414642, 4, 0.3333333333333333, 3]
        fn_model = SCRIPT_PATH + "/../data/lgbm_model.dat"

        predictor = LGBMPredictor(fn_model)
        score = predictor.eval(feature)
        self.assertAlmostEqual(score, 0.0956292553650266)

    def test_script(self):
        feature = [2, -0.2226236411773681, -0.8203237354755402, -8.7032166,
                   -8.387475499999999, 4, 0, 0.9999975000062501, -0.5301455855369568,
                   -6.073882102966309, -4.049746978282928, 69.62790210378144,
                   -19.428411155939102, -20.248734891414642, 4, 0.3333333333333333, 3]
        fn_model = SCRIPT_PATH + "/../data/lgbm_model.dat"

        predictor_op = matx.script(LGBMPredictor)(fn_model)
        score = predictor_op.eval(feature)
        self.assertAlmostEqual(score, 0.0956292553650266)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
