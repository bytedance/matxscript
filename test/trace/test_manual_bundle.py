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
import matx

from typing import Any


class MyBundleTest:

    def __init__(self, loc: str, norm_type: str):
        self.loc: str = loc
        self.norm_type: str = norm_type

    def __call__(self) -> Any:
        return 0


op = matx.script(MyBundleTest, bundle_args=['loc'])("model_0", "nfkc_0")


def process():
    m = op()
    return m


if __name__ == '__main__':
    save_path = "./test_back"
    m1 = matx.trace(process)
    m1.save(save_path)
    m2 = matx.load(save_path, "cpu")
    print(m2.run({}))
