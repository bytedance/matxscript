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

import builtins
import abc
import argparse
import array
import asyncio
import ast
import atexit
import base64
import binascii
import binhex
import bisect
import bz2
import collections
import copy
import csv
import configparser
import cmath
import ctypes
import datetime
import decimal
import functools
import gc
import getopt
import gzip
import hashlib
import html
import http
import importlib
import io
import ipaddress
import itertools
import inspect
import json
import logging
import math
import multiprocessing
import os
import pdb
import random
import re
import shutil
import sys
import string
import struct
import socket
import statistics
import threading
import types
import unittest
import urllib
import uuid
import typing
import unicodedata
import time
import timeit
import trace
import traceback
import xml
import zipfile
import zlib

BUILTIN_MODULES = [
    builtins,
    abc,
    argparse,
    array,
    ast,
    asyncio,
    atexit,
    base64,
    binascii,
    binhex,
    bisect,
    bz2,
    collections,
    configparser,
    cmath,
    copy,
    csv,
    ctypes,
    datetime,
    decimal,
    functools,
    gc,
    getopt,
    gzip,
    hashlib,
    html,
    http,
    importlib,
    io,
    ipaddress,
    itertools,
    inspect,
    json,
    logging,
    math,
    multiprocessing,
    os,
    pdb,
    random,
    re,
    shutil,
    sys,
    string,
    struct,
    socket,
    statistics,
    threading,
    types,
    unittest,
    urllib,
    uuid,
    typing,
    unicodedata,
    time,
    timeit,
    trace,
    traceback,
    threading,
    xml,
    zipfile,
    zlib,
]
