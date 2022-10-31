import os
import subprocess

CUDA_HOME= os.environ.get('CUDA_HOME', None)
CUDA_VERSION = None
MLSYS_COMPILE_CUDA_VERSION = os.environ.get('MLSYS_COMPILE_CUDA_VERSION', None)

def get_nvcc_version():
  global CUDA_HOME
  nvcc_bin = os.path.join(CUDA_HOME, 'bin/nvcc')
  get_nvcc_version_cmd = '{} --version |  grep -Po "release \\K([0-9]{{1,}}\\.)+[0-9]{{1,}}"'.format(nvcc_bin)
  nvcc_version = subprocess.check_output(get_nvcc_version_cmd, shell=True).decode('utf-8')
  nvcc_version = nvcc_version.strip()
  return nvcc_version

try:
  if MLSYS_COMPILE_CUDA_VERSION is not None:
    CUDA_VERSION = MLSYS_COMPILE_CUDA_VERSION
  if CUDA_VERSION is None:
    CUDA_VERSION = get_nvcc_version()
  print("CUDA_VERSION: {}".format(CUDA_VERSION))
  # set ENV needed by data/mlsys_deps
  if 'MLSYS_COMPILE_CUDA_VERSION' not in os.environ:
    os.environ['MLSYS_COMPILE_CUDA_VERSION']=CUDA_VERSION
  if 'MLSYS_COMPILE_CUDNN_VERSION' not in os.environ:
    if CUDA_VERSION >= '11.1':
      os.environ['MLSYS_COMPILE_CUDNN_VERSION']="8.2.0"
    elif CUDA_VERSION == '11.0':
      os.environ['MLSYS_COMPILE_CUDNN_VERSION']="8.0.4"
    else:
      os.environ['MLSYS_COMPILE_CUDNN_VERSION']="7.6.5"
except:
  print("[WARN] Failed to get cuda version.")


def get_cuda_deps():
  global CUDA_VERSION
  default_version = '10.1'

  if CUDA_VERSION is None:
    print("[INFO] use default_version {}.".format(default_version))
    return [
      "cpp3rdlib/cuda:10.1.243@//cpp3rdlib/cuda:cuda,cudart",
    ]
  else:
    return [
      "data/mlsys_deps:master@//data/mlsys_deps/cuda:cuda,cudart",
    ]

__CUDA_DEPS__ = get_cuda_deps()

__CPP_FLAGS__ = [
  "-std=c++14", "-O2", "-g", "-pthread", "-fPIC -w"
]

__CPP_FLAGS_DEBUG__ = [
  "-std=c++14", "-O0", "-g", "-pthread", "-fPIC -w"
]

# matx_inc will be used to build c++ extension.
cc_library(
  name="matx_inc",
  srcs=[],
  incs=[
    "include",
  ],
  defs=[
    "MATX_ENABLE_PCRE_REGEX=1",
  ],
  deps=[
    "cpp3rdlib/rapidjson:1.1.0@//cpp3rdlib/rapidjson:rapidjson",
    "matx/filesystem:1.5.0@//matx/filesystem:filesystem",
    "data/bfc-arena-allocator:master@//data/bfc-arena-allocator:bfc_arena_allocator_common",
  ],
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=True,
  bundle_path="lib",
)

cc_library(
  name="matx_runtime",
  srcs=[
    "src/runtime/*.cc",
    "src/runtime/*/*.cc",
    "src/runtime/*/*/*.cc",
    "src/pipeline/*.cc",
    "src/server/*.cc",
  ],
  exclude_srcs=[
    "src/runtime/cuda/*.cc",
  ],
  incs=[
    "include",
  ],
  defs=[
    "MATX_ENABLE_PCRE_REGEX=1",
  ],
  deps=[
    "#pthread",
    "#dl",
    "data/bfc-arena-allocator:master@//data/bfc-arena-allocator:bfc_arena_allocator_common",
    "cpp3rdlib/pcre:v8.43@//cpp3rdlib/pcre:pcre",
    "cpp3rdlib/rapidjson:1.1.0@//cpp3rdlib/rapidjson:rapidjson",
    "matx/filesystem:1.5.0@//matx/filesystem:filesystem",
  ],
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=False,
  link_all_symbols=True,
  bundle_path="lib",
)

cc_library(
  name="matx_runtime_cuda",
  srcs=[
    "src/runtime/cuda/*.cc",
  ],
  incs=[
    "include",
  ],
  deps=[
    ":matx_runtime",
    "data/bfc-arena-allocator:master@//data/bfc-arena-allocator:bfc_arena_allocator_cuda",
  ] + __CUDA_DEPS__,
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=False,
  link_all_symbols=True,
  bundle_path="lib",
)

cc_library(
  name="matx_runtime_torch",
  srcs=[
    "python/matx/extension/cpp/pytorch/src/*.cc",
  ],
  incs=[
    "include",
    "python/matx/extension/cpp/pytorch/src/",
  ],
  deps=[
    ":matx_runtime",
    "data/mlsys_deps:master@//data/mlsys_deps/frameworks/torch:torch", # can be override by real user
  ],
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=False,
  link_all_symbols=True,
  bundle_path="lib",
)

cc_library(
  name="matx_runtime_tensorflow_1",
  srcs=[
    "python/matx/extension/cpp/tensorflow/src/*.cc",
  ],
  incs=[
    "include",
    "python/matx/extension/cpp/tensorflow/src/",
  ],
  deps=[
    ":matx_runtime",
    "cpp3rdlib/tensorflow:1.15.3-cpu@//cpp3rdlib/tensorflow:tensorflow_framework,tensorflow_cc", # can be override by real user
  ],
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=False,
  link_all_symbols=True,
  bundle_path="lib",
)

cc_library(
  name="matx_runtime_tensorflow_2",
  srcs=[
    "python/matx/extension/cpp/tensorflow/src/*.cc",
  ],
  incs=[
    "include",
    "python/matx/extension/cpp/tensorflow/src/",
  ],
  deps=[
    ":matx_runtime",
    "cpp3rdlib/lab_tensorflow:2.5.0-pb3.9.2-cuda11.1-gcc8@//cpp3rdlib/lab_tensorflow:tensorflow_framework,tensorflow_cc", # can be override by real user
  ],
  custom_deps=[],
  extra_cppflags=__CPP_FLAGS__,
  export_incs=["include"],
  allow_undefined=False,
  link_all_symbols=True,
  bundle_path="lib",
)
