
nvcc -O2 -o test_cuda test_cuda.cc
g++ -o test_fork -O2 test_fork.cc -lpthread
nvcc -O2 -o test_fork_set_device test_fork_set_device.cc
