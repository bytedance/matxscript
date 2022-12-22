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
import multiprocessing as mp


@matx.script
def my_func(a: int) -> None:
    print("a: ", a)


def t():
    print("[multiprocessing] child begin", flush=True)
    matx.pmap(my_func, [1, 2, 3, 4])
    print("pmap_threads", matx.pipeline.TXObject.default_sess.get_pmap_threads())
    print("async_threads", matx.pipeline.TXObject.default_sess.get_apply_async_threads())
    print("[multiprocessing] child end", flush=True)


if __name__ == "__main__":
    print("begin test multiprocessing", flush=True)
    mp.set_start_method('fork')
    p = mp.Process(target=t)
    p.start()
    p.join()
    print("end test multiprocessing", flush=True)
