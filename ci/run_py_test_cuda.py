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
import subprocess
import multiprocessing


THIS_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(THIS_PATH, '..')

###############################################################################
# build all shared target
###############################################################################
os.chdir(ROOT_PATH)
if len(sys.argv) > 1 and sys.argv[1] == '--no_rebuild':
    pass
else:
    os.system('BUILD_TESTING=OFF BUILD_BENCHMARK=OFF CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" bash ci/build_lib.sh')

###############################################################################
# install requirements
###############################################################################
PYTHON_MODULE_PATH = os.path.join(ROOT_PATH, 'python')
os.chdir(PYTHON_MODULE_PATH)
os.system('pip3 install -r requirements.txt')

###############################################################################
# find all test script
###############################################################################
TEST_SCRIPT_PATH = os.path.join(ROOT_PATH, 'test/script/cuda')
os.chdir(TEST_SCRIPT_PATH)

test_files = []
for a, b, c in os.walk('./'):
    for cc in c:
        if cc.startswith('test_') and cc.endswith('.py'):
            test_files.append(cc)

PYTHONPATH = os.path.join(ROOT_PATH, 'python')
os.environ['PYTHONPATH'] = PYTHONPATH + ':' + os.environ.get('PYTHONPATH', '')


###############################################################################
# run all test script
###############################################################################
def worker(test_file):
    env = os.environ.copy()
    env['MATX_DSO_DIR'] = test_file + '_dso'
    proc = subprocess.Popen(['python3', test_file], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    return test_file, proc.returncode, out


failed = []
cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(int(cpus * 1.5)) as pool:
    it = pool.imap_unordered(worker, test_files)

    for ret in it:
        if ret[1] != 0:
            failed.append(ret[0])
            print('+' * 40)
            print(ret[0], 'failed!!!')
            print(ret[2].decode())
            print('+' * 40)
        else:
            print(ret[0], 'passed.')

print('{} test files in total, {} passed.'.format(len(test_files), len(test_files) - len(failed)))
if failed:
    print('Following files failed: ', failed)
    sys.exit(1)
