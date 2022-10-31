// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

#include <cuda_runtime.h>

#define CUDA_CALL(func)                                         \
  {                                                             \
    cudaError_t e = (func);                                     \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) { \
      fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));     \
    }                                                           \
  }

void SetDevice(int device_id) {
  printf("begin cudaSetDevice(%d)...\n", device_id);
  CUDA_CALL(cudaSetDevice(device_id));
  printf("end cudaSetDevice(%d)\n", device_id);
}

void checker() {
  SetDevice(0);
}

void child_entry(int sleep_ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  printf("---------------------------------------------------------------\n");
  printf("Child: My PID= %d, parent= %d\n", (int)getpid(), (int)getppid());
  checker();
  printf("---------------------------------------------------------------\n\n");
}

void parent_entry() {
  printf("---------------------------------------------------------------\n");
  printf("Parent: My PID= %d\n", (int)getpid());
  checker();
  printf("---------------------------------------------------------------\n\n");
}

pid_t create_process(int sleep_ms) {
  pid_t pid = fork();

  if (pid < 0) {
    fprintf(stderr, "IPT: fork() error\n");
    return -1;
  } else {
    if (pid == 0) {
      child_entry(sleep_ms);
      exit(0);
    } else {
      return pid;
    }
  }
}

void join_pid(pid_t pid) {
  int status;
  if (waitpid(pid, &status, 0) == -1) {
    fprintf(stderr, "Parent: wait() error");
  } else if (WIFEXITED(status)) {
    printf("Child exited with status: %d\n", WEXITSTATUS(status));
  } else {
    printf("Child did not exit successfully\n");
  }
}

int main(int argc, char* argv[]) {
  int before = 0;
  int process_num = 2;
  pid_t* pid_list = nullptr;

  if (argc == 2) {
    before = std::atoi(argv[1]);
  } else if (argc == 3) {
    auto num = std::atoi(argv[2]);
    if (num > 0) {
      process_num = num;
    }
  }

  if (before) {
    printf("warmup in parent process...\n");
    parent_entry();
  }

  pid_list = new pid_t[process_num];

  std::random_device rd;                                 // obtain a random number from hardware
  std::mt19937 gen(rd());                                // seed the generator
  std::uniform_int_distribution<> random_dist(1, 1000);  // define the range

  printf("create process...\n");
  for (auto i = 0; i < process_num; ++i) {
    pid_list[i] = create_process(random_dist(gen));
  }
  sleep(1);
  printf("join...\n");
  for (auto i = 0; i < process_num; ++i) {
    join_pid(pid_list[i]);
  }
  parent_entry();
  delete[] pid_list;
  return 0;
}
