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
import matx
from typing import Any

cpu_device = matx.Device("cpu")
gpu_device = matx.Device("gpu:0")
default_device = matx.Device("")


class PrefixDeviceOp:
    def __init__(self, device: Any = None) -> None:
        self.func: Any = matx.make_native_object("MyDeviceOpExample", device())

    def __call__(self, prefix: bytes) -> bytes:
        return self.func.device_check(prefix)


class TestDeviceOpDict(unittest.TestCase):
    def test_python_use(self):
        # none_device_op = PrefixDeviceOp()
        cpu_device_op = PrefixDeviceOp(cpu_device)
        gpu_device_op = PrefixDeviceOp(gpu_device)
        # self.assertEqual(b"haha:-32768:-32768", none_device_op(b"haha"))
        self.assertEqual(b"haha:0:-32768", gpu_device_op(b"haha"))
        self.assertEqual(b"haha:-1:-32768", cpu_device_op(b"haha"))

    def test_script_use(self):
        creator = matx.script(PrefixDeviceOp)
        # none_device_op = creator()
        gpu_device_op = creator(gpu_device)
        cpu_device_op = creator(cpu_device)
        # self.assertEqual(b"haha:-32768:-32768", none_device_op("haha"))
        self.assertEqual(b"haha:0:-32768", gpu_device_op(b"haha"))
        self.assertEqual(b"haha:-1:-32768", cpu_device_op(b"haha"))

    def test_session_use(self):
        creator = matx.script(PrefixDeviceOp)
        # none_device_op = creator()
        gpu_device_op = creator(gpu_device)
        cpu_device_op = creator(cpu_device)

        def process(prefix):
            gpu_prefix = gpu_device_op(prefix)
            cpu_prefix = cpu_device_op(prefix)
            return gpu_prefix, cpu_prefix

        gpu_prefix, cpu_prefix = process(b"haha")
        self.assertEqual(b"haha:0:-32768", gpu_prefix)
        self.assertEqual(b"haha:-1:-32768", cpu_prefix)

        session = matx.pipeline.Trace(process, b"haha")
        session.save("device_op_session")
        t_session = matx.pipeline.Load("device_op_session", 1)
        gpu_prefix, cpu_prefix = t_session.run({"prefix": b"xixi"})
        self.assertEqual(b"xixi:0:1", gpu_prefix)
        self.assertEqual(b"xixi:-1:1", cpu_prefix)

    def test_script_device(self):
        def _make_device() -> Any:
            d: matx.Device = matx.Device("cpu")
            c: matx.pipeline.ops.DeviceOp = matx.Device("cpu")
            return d, c

        matx.script(_make_device)()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
