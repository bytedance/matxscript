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
from typing import List, Dict, Callable, Any, AnyStr

import matx


class Text2Ids:
    def __init__(self, texts: List[str]) -> None:
        self.table: Dict[str, int] = {}
        for i in range(len(texts)):
            self.table[texts[i]] = i

    def __call__(self, words: List[str]) -> List[int]:
        return [self.table.get(word, -1) for word in words]


if __name__ == "__main__":
    ##  Basic usage
    op = Text2Ids(["hello", "world"])
    examples = "hello world unknown".split()
    ret = op(examples)
    print(ret)
    # should print out [0, 1, -1]

    ##  Script
    script_op = matx.script(Text2Ids)(["hello", "world"])
    ret = script_op(examples)
    print(ret)
    # should print out [0, 1, -1]

    ##  Trace
    def wrapper(inputs):
        return script_op(inputs)
    # trace and save
    traced = matx.trace(wrapper, examples)
    traced.save("demo_text2id")
    # load and run
    # for matx.load，the first argument is the stored trace path
    # the second argument indicates the device for further running the code
    # -1 means cpu, if for gpu, just pass in the device id
    loaded = matx.load("demo_text2id", -1)
    # we call 'run' interface here to actually run the traced op
    # note that the argument is a dict, where the key is the arg name of the traced function
    # and the value is the actual input data
    ret = loaded.run({"inputs": examples})
    print(ret)
    # should print out [0, 1, -1]

