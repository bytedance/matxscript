# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disable CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA OFF)


set(BUILD_TESTING OFF)
set(BUILD_BENCHMARK OFF)
set(USE_LIBBACKTRACE ON)
set(CPPFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")