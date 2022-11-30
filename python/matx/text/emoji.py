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
import sys
from typing import List, Callable, AnyStr, Any

from ..native import make_native_object
from ._dso_loader import load_text_ops_lib

load_text_ops_lib()
matx = sys.modules['matx']


class EmojiFilter(object):

    def __init__(self,
                 common_unicode: bool = True,
                 common_unicode_trans: bool = False,
                 common_unicode_trans_alias: bool = False,
                 user_emojis: Any = None,
                 ) -> None:
        self.filter_engine: Any = make_native_object(
            "text_emoji_EmojiFilter",
            common_unicode,
            common_unicode_trans,
            common_unicode_trans_alias,
            user_emojis,
        )

    def filter(self, s: AnyStr) -> AnyStr:
        return self.filter_engine.filter(s)

    def replace(self, s: AnyStr, repl: AnyStr, keep_all: bool) -> AnyStr:
        return self.filter_engine.replace(s, repl, keep_all)

    def check_pos(self, s: AnyStr, pos: int) -> int:
        return self.filter_engine.check_pos(s, pos)
