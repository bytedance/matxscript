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

import math
from typing import Dict, List, Tuple

from ..typing import FTList


def sigmoid(x: float) -> float:
    if math.isinf(x) or math.isnan(x):
        return 0.
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))


class LgbmModelTree:
    def __init__(self) -> None:
        self.split_feature: FTList[int] = []
        self.left_child: FTList[int] = []
        self.right_child: FTList[int] = []
        self.threshold: FTList[float] = []
        self.leaf_value: FTList[float] = []


class LGBMPredictor:
    def __init__(self, fn_model: str) -> None:
        self.trees: FTList[LgbmModelTree] = []
        fin = open(fn_model)
        tree_lines = list()
        while True:
            tree_lines.clear()
            while True:
                line = fin.readline()
                if line == "":
                    break
                if 'Tree=' in line:
                    line = fin.readline()
                    while line.strip() != '':
                        tree_lines.append(line)
                        line = fin.readline()
                    break
            if len(tree_lines) == 0:
                break
            self.trees.append(self._build_tree(tree_lines))
        fin.close()

    def eval(self, data: List[float]) -> float:
        # Can't be compiled
        # flat_data: FTList[float] = list(data)
        flat_data: FTList[float] = [v for v in data]
        s = 0.0
        for tree in self.trees:
            node = 0
            while node >= 0:
                if flat_data[tree.split_feature[node]] <= tree.threshold[node]:
                    node = tree.left_child[node]
                else:
                    node = tree.right_child[node]
            node = ~node
            s += tree.leaf_value[node]
        return sigmoid(s)

    def predict(self, data_list: List[List[float]]) -> List[float]:
        ret: List[float] = []
        for data in data_list:
            ret.append(self.eval(data))
        return ret

    def _build_tree(self, tree_lines: List[str]) -> LgbmModelTree:
        ret = LgbmModelTree()
        for line in tree_lines:
            key, value = line.strip().split('=')
            values = value.split(' ')
            if key == 'split_feature':
                for v in values:
                    ret.split_feature.append(int(v))
            elif key == 'left_child':
                for v in values:
                    ret.left_child.append(int(v))
            elif key == 'right_child':
                for v in values:
                    ret.right_child.append(int(v))
            elif key == 'threshold':
                for v in values:
                    ret.threshold.append(float(v))
            elif key == 'leaf_value':
                for v in values:
                    ret.leaf_value.append(float(v))
        return ret

    def __call__(self, data_list: List[List[float]]) -> List[float]:
        return self.predict(data_list)
