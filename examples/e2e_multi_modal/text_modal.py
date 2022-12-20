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
from typing import List, Dict, Any


class TextCleaner:
    """TextCleaner impl by matx."""

    def __init__(self) -> None:
        self.white_regex: matx.Regex = matx.Regex(r"[ \t\n\r\p{Zs}]")
        self.control_regex: matx.Regex = matx.Regex(
            r"[\u0000\ufffd\p{Cc}\p{Cf}\p{Mn}]")

        self.space: bytes = " ".encode()
        self.empty: bytes = "".encode()

    def __call__(self, text: bytes) -> bytes:
        t = self.white_regex.replace(text, self.space)
        return self.control_regex.replace(t, self.empty)


class CaseNormalizer:
    def __init__(self, do_lowercase: bool = False, unicode_norm: str = '') -> None:
        self.do_lowercase: bool = do_lowercase

    def __call__(self, text: bytes) -> bytes:
        if self.do_lowercase:
            return text.lower()
        else:
            return text


class PunctuationPadding:
    """Pad a space around the punctuation."""

    def __init__(self):
        self.regex_pattern: matx.Regex = matx.Regex(
            r"([\u0021-\u002f]|[\u003a-\u0040}]|[\u005b-\u0060}]|[\u007b-\u007e]|\p{P})")
        self.replace_pattern: bytes = r" ${1} ".encode()

    def __call__(self, text: bytes) -> bytes:
        return self.regex_pattern.replace(text, self.replace_pattern)


class MatxBertTokenizer:
    def __init__(self,
                 vocab_path: str,
                 lower_case: bool = False,
                 max_tokens_per_input: int = 256,
                 unk_token: str = '[UNK]'
                 ) -> None:
        """
        matx style BertTokenzier。
        vocab_path: vocabulary path for tokenizer
        lower_case: convert to lowercase or not
        max_tokens_per_input: token length limit
        unk_token: the symbol for unknown tokens
        """
        self.cleaner: TextCleaner = TextCleaner()
        self.normalizer: CaseNormalizer = CaseNormalizer(True)
        self.punc_padding: PunctuationPadding = PunctuationPadding()
        self.max_tokens_per_input: int = max_tokens_per_input
        self.world_piece: Any = WordPieceTokenizer(vocab_path=vocab_path,
                                                   unk_token=unk_token,
                                                   max_bytes_per_token=max_tokens_per_input)
        self.cls_id: int = self.world_piece.tokenize(['[CLS]'])[0]
        self.sep_id: int = self.world_piece.tokenize(['[SEP]'])[0]
        self.pad_id: int = self.world_piece.tokenize(['[PAD]'])[0]

    def __call__(self, texts: List[bytes]) -> Dict[str, matx.NDArray]:
        batch_input_ids: List = []
        batch_input_mask: List = []
        batch_segment_ids: List = []
        for text in texts:
            text = self.cleaner(text)
            text = self.normalizer(text)
            text = self.punc_padding(text)
            terms: List = text.split()
            tokens: List[int] = self.world_piece.tokenize(terms)
            # start to create bert style input
            len_tre: int = self.max_tokens_per_input - 2
            input_ids: List = [self.cls_id] + tokens[:len_tre] + [self.sep_id]
            input_mask: List = [1] * len(input_ids) + [0] * \
                (self.max_tokens_per_input - len(input_ids))
            input_ids = input_ids + [self.pad_id] * (self.max_tokens_per_input - len(input_ids))
            segment_ids = [0] * self.max_tokens_per_input
            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_segment_ids.append(segment_ids)
        res: Dict = {}
        res["input_ids"] = matx.NDArray(batch_input_ids, [], "int64")
        res["input_mask"] = matx.NDArray(batch_input_mask, [], "int64")
        res["segment_ids"] = matx.NDArray(batch_segment_ids, [], "int64")
        return res
