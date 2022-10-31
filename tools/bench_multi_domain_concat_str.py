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

from typing import List

import time
import prettytable
import matx
import tabulate
import torch
import tensorflow as tf

Loops = 1000


def multi_domain_concat(query_tok: List[str],
                        domains: List[List[str]],
                        skip_empty: bool = True) -> List[str]:
    # [CLS] query [SEP] title [SEP] username [SEP] ... [SEP]
    inputs: List[str] = []
    segment_ids: List[int] = []

    inputs.append('[CLS]')
    segment_ids.append(0)
    for q_tok in query_tok:
        inputs.append(q_tok)
        segment_ids.append(0)
    inputs.append('[SEP]')
    segment_ids.append(0)

    seg_id = 1
    for domain in domains:
        domain_size = len(domain)
        if domain_size == 0 and skip_empty:
            continue
        for sub_d in domain:
            inputs.append(sub_d)
            segment_ids.append(seg_id)
        seg_id += 1
        inputs.append('[SEP]')

    return inputs


def prepare_data(mode=None):
    query_tok = ['abc', 'bbc', 'cd']
    domains = [['da', 'ef', 'fw'] * 10, ['gd', 'hg', 'ih'] * 10, ['ju', 'kk', 'lw'] * 10]
    skip_empty = True
    if mode == 'matx':
        return matx.List(query_tok), matx.List(domains), skip_empty
    # elif mode == 'tf':
    #     return tf.ragged.constant(query_tok), tf.ragged.constant(domains), skip_empty
    # elif mode == 'torch':
    #     return query_tok, domains, skip_empty
    else:
        return query_tok, domains, skip_empty


def get_exe_time(func_impl, mode=None):
    # test case
    query_tok, domains, skip_empty = prepare_data(mode=mode)
    try:
        for i in range(300):
            func_impl(query_tok, domains, skip_empty)
        begin = time.time()
        func_impl(query_tok, domains, skip_empty)
        end = time.time()
        return "%.4f(ms)" % ((end - begin) * 1000)
    except:
        import traceback
        traceback.print_exc()
        return "-"


def test_multi_fields_concat():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    field_names = ["framework", "version", "loops", "time(ms)"]
    result_info = [
        ["python", "3.7", Loops, get_exe_time(multi_domain_concat)],
        ["PyTorch", "1.6", Loops, get_exe_time(torch.jit.script(multi_domain_concat), mode="torch")],
        ["TensorFlow", "2.4", Loops, get_exe_time(tf.function(multi_domain_concat), mode="tf")],
        ["MatxScript", "1.6", Loops, get_exe_time(matx.script(multi_domain_concat), mode="matx")],
    ]
    output_tb = prettytable.PrettyTable()
    output_tb.field_names = field_names
    output_tb.add_rows(result_info)
    # print(output_tb.get_string())

    output_tb = tabulate.tabulate(
        result_info, headers=output_tb.field_names, tablefmt="rst"
    )
    print(output_tb)


def build_compasion_table():
    output_tb = tabulate.tabulate(
        [
            ["Python", "3.7", "✗", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
            ["matx4", "1.2", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✗", "✗"],
            ["PyTorch", "1.6", "✓", "✓", "✗", "✓", "✓", "?", "✗", "✓", "✓"],
            ["TensorFlow", "2.4", "✓", "✗", "✓", "✗", "✗", "✗", "✗", "✓", "✓"],
            ["TVM", "0.7", "✓", "✗", "✗", "✓", "✗", "✗", "✗", "✓", "?"],
            ["pypy", "3.7", "✗", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✗"],
        ], headers=["framework", "version", "run without python", "str", "bytes", "list", "dict",
                    "iterator", "class",
                    "tensor", "autodiff"], tablefmt="rst"
    )
    print(output_tb)


test_multi_fields_concat()
# build_compasion_table()
