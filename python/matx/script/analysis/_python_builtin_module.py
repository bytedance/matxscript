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
import os
import sys

BUILTIN_MODULE_NAMES = [
    "builtins",
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "atexit",
    "base64",
    "binascii",
    "binhex",
    "bisect",
    "bz2",
    "collections",
    "configparser",
    "cmath",
    "copy",
    "csv",
    "ctypes",
    "datetime",
    "decimal",
    "functools",
    "gc",
    "getopt",
    "gzip",
    "hashlib",
    "html",
    "http",
    "importlib",
    "io",
    "ipaddress",
    "itertools",
    "inspect",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "os",
    "pdb",
    "random",
    "re",
    "shutil",
    "sys",
    "string",
    "struct",
    "socket",
    "statistics",
    "threading",
    "types",
    "unittest",
    "urllib",
    "uuid",
    "typing",
    "unicodedata",
    "time",
    "timeit",
    "trace",
    "traceback",
    "threading",
    "xml",
    "zipfile",
    "zlib",
]


def __get_python_builtin_module_path():
    site = sys.modules['site']
    mod_path = os.path.abspath(os.path.split(os.path.realpath(site.__file__))[0])
    return mod_path, site.getsitepackages()[0]


BUILTIN_MODULE_PATH, SITE_PACKAGES_PATH = __get_python_builtin_module_path()


def is_builtin_module(mod):
    mod_name = mod.__name__
    if mod_name in sys.builtin_module_names:
        return True
    mod_file = getattr(mod, "__file__", None)
    if mod_file is None:
        return True
    if mod_name in BUILTIN_MODULE_NAMES:
        mod_path = os.path.abspath(os.path.split(os.path.realpath(mod_file))[0])
        if mod_path.startswith(BUILTIN_MODULE_PATH) and not mod_path.startswith(SITE_PACKAGES_PATH):
            return True
    return False
