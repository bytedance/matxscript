import os
import sys
import subprocess


THIS_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(THIS_PATH, '..')

###############################################################################
# find all test script
###############################################################################
TEST_SCRIPT_PATH = os.path.join(ROOT_PATH, 'test/vision')
os.chdir(TEST_SCRIPT_PATH)

test_files = []
for a, b, c in os.walk('./'):
    for cc in c:
        if cc.startswith('test_') and cc.endswith('.py'):
            test_files.append(os.path.join(a, cc))

PYTHONPATH = os.path.join(ROOT_PATH, 'python')
os.environ['PYTHONPATH'] = PYTHONPATH + ':' + os.environ.get('PYTHONPATH', '')

###############################################################################
# run all test script
###############################################################################
failed = []


def worker(test_file):
    env = os.environ.copy()
    proc = subprocess.Popen(['python3', test_file], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    return test_file, proc.returncode, out


for test_file in test_files:
    _, code, out = worker(test_file)
    if code != 0:
        failed.append(test_file)
        print('+' * 40)
        print(test_file, 'failed!!!')
        print(out.decode())
        print('+' * 40)
    else:
        print(test_file, 'passed.')

print('{} test files in total, {} passed.'.format(
    len(test_files), len(test_files) - len(failed)))
if failed:
    print('Following files failed: ', failed)
    sys.exit(1)
