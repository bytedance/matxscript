# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: This file originates from incubator-tvm
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
import re
import logging
import traceback
import builtins

from ..env import MATX_DEV_MODE
from ..contrib.statistic import counter


def _matx_codegen_demangle(fname):
    c_api = '__c_api'
    if fname.endswith(c_api):
        fname = fname[:-len(c_api)]
    tokens = fname.split('__F_')
    if len(tokens) != 2:
        return fname
    return tokens[1]


def _strip_namespace_and_args(fname):
    namespace = "(anonymous namespace)::"
    pos_l = 0
    pos_r = None
    if fname.startswith(namespace):
        pos_l = len(namespace)
    pos = fname.find('(', pos_l, pos_r)
    if pos != -1:
        pos_r = pos
    pos = fname.find('::', pos_l, pos_r)
    if pos != -1:
        pos_l = pos + 2
    return fname[pos_l:pos_r]


def _trace_to_python(cc_stack):
    stack_filter = set()
    file_name_to_contents = {}
    # cache python files
    for cc_frame in cc_stack:
        if cc_frame[0]:
            _, func, file_name, lineno = cc_frame[1]
            if file_name in file_name_to_contents:
                continue
            with open(file_name, 'r') as fr:
                file_name_to_contents[file_name] = list(fr)
    # translate stack
    ss = ['\n']
    for cc_frame in reversed(cc_stack):
        _, func, file_name, lineno = cc_frame[1]
        if not cc_frame[0]:
            frame = '  File "{}", line {}, in {}\n'.format(file_name, lineno, func)
            ss.append(frame)
            continue
        codeline = file_name_to_contents[file_name][lineno - 1]
        tokens = codeline.split('//')
        if len(tokens) < 2:  # format error, lack of lineno info.
            return None
        piece = tokens[-1]
        if piece in stack_filter:
            continue
        stack_filter.add(piece)
        py_span = piece.strip().split(':')
        func = _matx_codegen_demangle(func)
        frame = '  File "{}", line {}, in {}\n'.format(py_span[0], py_span[1], func)
        ss.append(frame)
        # TODO(maxiandi): getcode
        ss.append('    \n')
    return ss


def _trans_message(what):
    if isinstance(what, Exception):
        what = str(what)
    pos = what.find("Stack trace:\n")
    if pos < 0:
        return what, "", ""

    top = what[0:pos]
    cc_stack_info = what[pos:]
    lines = cc_stack_info.split('\n')
    # print(lines)
    i = 0
    cc_stack = []
    bt_pat_header = re.compile(r"^(\d+):\s+(.*)\s*$")
    bt_pat_body = re.compile(r"^\s+at\s*(.*?):(\d+)$")
    bt_pat_py_code = re.compile(r".*dso/lib.*_plugin_[a-zA-Z\d]+\.cc$")
    bt_pat_runtime = re.compile(r".*matxscript/src.*")
    while i < len(lines):
        line = lines[i].strip()
        matched = bt_pat_header.match(line)
        if matched:
            frame = int(matched.group(1))
            funcname = matched.group(2)
            info = lines[i + 1]
            info_matched = bt_pat_body.match(info)
            if info_matched:
                filename = info_matched.group(1)
                lineno = info_matched.group(2)
                if bt_pat_py_code.match(filename):
                    # codegen stack
                    funcname = _strip_namespace_and_args(funcname)
                    cc_stack.append((1, (frame, funcname, filename, int(lineno))))
                elif bt_pat_runtime.match(filename):
                    # runtime stack
                    cc_stack.append((0, (frame, funcname, filename, int(lineno))))
                i += 1
        i += 1
    if len(cc_stack) == 0:
        return top, cc_stack_info, ""
    py_stack = _trace_to_python(cc_stack)
    if not py_stack:
        return top, cc_stack_info, ""
    if py_stack[-2].lstrip() in top:
        py_stack.pop()
        py_stack.pop()
    return top, cc_stack_info, ''.join(py_stack)


def trans_message(what):
    try:
        return _trans_message(what)
    except:
        logging.warn(
            "Failed to transform exception from c++ to python, reason:{}".format(traceback.format_exc()))
        return what, "", ""


def trans_exception(e: Exception, use_cc_stacktrace: bool):
    trans_state = getattr(e, "__IS_MATX_TRANS_EXCEPTION__", False)
    if trans_state:
        return e, trans_state
    top_pat = re.compile(r'^(File ".*\.*", line \d+.*?\n+)(.*)$')
    top, cc_origin_stack, py_stack = trans_message(str(e))
    stack_msg = cc_origin_stack if use_cc_stacktrace else py_stack
    full_msg = stack_msg + top  # ignore cc_origin_stack
    ty_pos = top.find(":")
    if ty_pos != -1:
        ex_ty_s = top[:ty_pos]
        ex_ty = getattr(builtins, ex_ty_s, type(e))
        if issubclass(ex_ty, BaseException):
            top = top[ty_pos + 1:].lstrip(' ')
            top_matched = top_pat.match(top)
            if top_matched:
                top = top_matched.group(1) + ex_ty_s.strip() + ": " + top_matched.group(2)
            msg = stack_msg + '  ' + top
            py_exc = ex_ty(msg.strip())
            py_exc.__IS_MATX_TRANS_EXCEPTION__ = True
            return py_exc, trans_state
    py_exc = type(e)(full_msg.strip())
    py_exc.__IS_MATX_TRANS_EXCEPTION__ = True
    return py_exc, trans_state


def trans_exception_from_c_to_py(f):
    if MATX_DEV_MODE:
        return f

    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            ex, trans_state = trans_exception(e, use_cc_stacktrace=False)
            counter.set('matx_runtime_error', str(ex))
            counter.set('matx_runtime_error_counter', 1)
            counter.flush()
            raise ex from None

    return wrap
