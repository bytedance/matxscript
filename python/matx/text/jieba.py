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
"""
supported global functions in jieba(py)

[ ] get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
[ ] add_word = dt.add_word
[ ] calc = dt.calc
[*] cut = dt.cut
[*] lcut = dt.lcut
[*] cut_for_search = dt.cut_for_search
[*] lcut_for_search = dt.lcut_for_search
[ ] del_word = dt.del_word
[ ] get_DAG = dt.get_DAG
[ ] get_dict_file = dt.get_dict_file
[ ] initialize = dt.initialize
[ ] load_userdict = dt.load_userdict
[ ] set_dictionary = dt.set_dictionary
[ ] suggest_freq = dt.suggest_freq
[ ] tokenize = dt.tokenize
[ ] user_word_tag_tab = dt.user_word_tag_tab
"""
import os
import sys
from typing import List, Tuple, AnyStr, Any

from ..native import make_native_object
from ._dso_loader import load_text_ops_lib

load_text_ops_lib()
matx = sys.modules['matx']

DICT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "jieba_dict"))


class JiebaImpl(object):

    def __init__(self,
                 dict_path: AnyStr = DICT_PATH + "/jieba.dict.utf8",
                 model_path: AnyStr = DICT_PATH + "/hmm_model.utf8",
                 user_dict_path: AnyStr = DICT_PATH + "/user.dict.utf8",
                 idfPath: AnyStr = DICT_PATH + "/idf.utf8",
                 stopWordPath: AnyStr = DICT_PATH + "/stop_words.utf8",
                 ) -> None:
        self.jieba: Any = make_native_object(
            "text_cutter_CPPJieba",
            dict_path,
            model_path,
            user_dict_path,
            idfPath,
            stopWordPath,
        )

    def __call__(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        return self.lcut(sentence, cut_all, HMM)

    def cut(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        """
        Generator is not supported now, `cut` behaves the same as `lcut`.
        """
        return self.jieba.lcut(sentence, cut_all, HMM)

    def lcut(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        return self.jieba.lcut(sentence, cut_all, HMM)

    def cut_for_search(self, sentence: AnyStr, HMM: bool = True) -> List[AnyStr]:
        """
        Generator is not supported now, `cut_for_search` behaves the same as `lcut_for_search`.
        """
        return self.jieba.lcut_for_search(sentence, HMM)

    def lcut_for_search(self, sentence: AnyStr, HMM: bool = True) -> List[AnyStr]:
        return self.jieba.lcut_for_search(sentence, HMM)


class Jieba:

    def __init__(self,
                 dict_path: AnyStr = DICT_PATH + "/jieba.dict.utf8",
                 model_path: AnyStr = DICT_PATH + "/hmm_model.utf8",
                 user_dict_path: AnyStr = DICT_PATH + "/user.dict.utf8",
                 idfPath: AnyStr = DICT_PATH + "/idf.utf8",
                 stopWordPath: AnyStr = DICT_PATH + "/stop_words.utf8",
                 ) -> None:
        self.jieba_op: Any = matx.script(JiebaImpl)(
            dict_path,
            model_path,
            user_dict_path,
            idfPath,
            stopWordPath,
        )

    def __call__(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        return self.jieba_op(sentence, cut_all, HMM)

    def cut(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        """
        Generator is not supported now, `cut` behaves the same as `lcut`.
        """
        return self.jieba_op.cut(sentence, cut_all, HMM)

    def lcut(self, sentence: AnyStr, cut_all: bool = False, HMM: bool = True) -> List[AnyStr]:
        return self.jieba_op.lcut(sentence, cut_all, HMM)

    def cut_for_search(self, sentence: AnyStr, HMM: bool = True) -> List[AnyStr]:
        """
        Generator is not supported now, `cut_for_search` behaves the same as `lcut_for_search`.
        """
        return self.jieba_op.cut_for_search(sentence, HMM)

    def lcut_for_search(self, sentence: AnyStr, HMM: bool = True) -> List[AnyStr]:
        return self.jieba_op.lcut_for_search(sentence, HMM)
