// alloc.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <ctime>
#include <iostream>

#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

//#define ENABLE_CUDA

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "core/device/cuda/cuda_allocator.h"
#endif

using namespace ::std;
using namespace ::brt;

bool checkBytes(void* ptr, char val, size_t size) {
  char* cPtr = (char*)ptr;
  for (int i = 0; i < size; ++i) {
    if (cPtr[i] != val)
      return false;
  }
  return true;
}

#ifdef ENABLE_CUDA
bool checkCUDABytes(void* ptr, char val, size_t size) {
  char* cPtr = (char*)malloc(size);
  cudaMemcpy(cPtr, ptr, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    if (cPtr[i] != val)
      return false;
  }

  free(cPtr);
  return true;
}
#endif

int main() {
  cout << "Alloc Testing!\n";

  std::clock_t start;
  double duration = 0;

  int cnt = 1024 * 32;
  size_t size = 1024;
  size_t cudaAlign = 128;

  cout << "test 1-" << cnt << " times allocation" << endl;

  std::cout << "=======================================\n";
  // test CPU allocator
  {
    cout << "test CPU base allocator\n";

    CPUAllocator baseAlloc;  // default CPU

    auto ptr = baseAlloc.Alloc(size);

    if (ptr == nullptr) {
      cout << "alloc failed\n";
      return -1;
    }

    // test the bytes are ok for read/write
    memset(ptr, -1, size);

    // check
    if (!checkBytes(ptr, -1, size)) {
      cout << "access failed\n";
      return -1;
    }

    baseAlloc.Free(ptr);

    cout << "test many allocation\n";
    // check time for many allocation
    std::vector<void*> ptrs;
    start = std::clock();
    for (int s = 1; s <= cnt; ++s) {
      void* raw = baseAlloc.Alloc(s);
      ptrs.push_back(raw);
    }

    duration = (std::clock() - start);
    cout << "alloc time: " << duration << '\n';

    start = std::clock();
    for (int s = 0; s < cnt; ++s) {
      baseAlloc.Free(ptrs[s]);
    }
    duration = (std::clock() - start);
    cout << "free time: " << duration << '\n';

    std::cout << "test CPU base allocator done\n";
  }

  std::cout << "=======================================\n";

  {
    cout << "test CPU bfc arena allocator\n";

    start = std::clock();
    BFCArena bfcAlloc(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 31);
    duration = (std::clock() - start);
    cout << "initial time: " << duration << '\n';

    // test alloc and access
    for (int s = 1; s <= cnt; ++s) {
      void* raw = bfcAlloc.Alloc(s * 128);
      if (raw == nullptr) {
        cout << "alloc failed\n";
        return -1;
      }

      memset(raw, -1, s);

      // check
      if (!checkBytes(raw, -1, s)) {
        cout << "access failed\n";
        return -1;
      }

      bfcAlloc.Free(raw);
    }

    // check time for many allocation
    cout << "test many allocation\n";
    std::vector<void*> ptrs;
    start = std::clock();
    for (int s = 1; s <= cnt; ++s) {
      void* raw = bfcAlloc.Alloc(s);
      ptrs.push_back(raw);
    }
    duration = (std::clock() - start);
    cout << "alloc time: " << duration << '\n';

    start = std::clock();
    for (int s = 0; s < cnt; ++s) {
      bfcAlloc.Free(ptrs[s]);
    }
    duration = (std::clock() - start);
    cout << "free time: " << duration << '\n';

    std::cout << "test CPU bfc arena allocator done\n";
  }

#ifdef ENABLE_CUDA
  std::cout << "=======================================\n";

  // test CUDA allocator
  {
    cout << "test CUDA base allocator\n";

    cudaSetDevice(0);
    CUDAAllocator baseAlloc(0, "cuda");  // default CPU

    auto ptr = baseAlloc.Alloc(size);

    if (ptr == nullptr) {
      cout << "alloc failed\n";
      return -1;
    }

    // test the bytes are ok for read/write
    cudaMemset(ptr, -1, size);

    // check
    if (!checkCUDABytes(ptr, -1, size)) {
      cout << "access failed\n";
      return -1;
    }

    baseAlloc.Free(ptr);

    // check time for many allocation
    cout << "test many allocation\n";
#if 0
        std::vector<void*> ptrs;
        start = std::clock();
        for (int s = 1; s <= cnt; ++s) {
            void* raw = baseAlloc.Alloc(s* cudaAlign);
            ptrs.push_back(raw);
        }

        duration = (std::clock() - start);
        cout << "alloc time: " << duration << '\n';

        start = std::clock();
        for (int s = 0; s < cnt; ++s) {
            baseAlloc.Free(ptrs[s]);
        }
        duration = (std::clock() - start);
        cout << "free time: " << duration << '\n';
#endif

    start = std::clock();
    for (int s = 1; s <= cnt; ++s) {
      void* raw = baseAlloc.Alloc(size);
      baseAlloc.Free(raw);
    }
    duration = (std::clock() - start);
    cout << "alloc/free time: " << duration << '\n';

    std::cout << "test CUDA base allocator done\n";
  }

  std::cout << "=======================================\n";

  {
    cout << "test CUDA bfc arena allocator\n";

    start = std::clock();
    BFCArena bfcAlloc(std::unique_ptr<IAllocator>(new CUDAAllocator(0, "cuda")), 1 << 31);
    duration = (std::clock() - start);
    cout << "initial time: " << duration << '\n';

    // test alloc and access
    for (int s = 1; s <= cnt; ++s) {
      void* raw = bfcAlloc.Alloc(s);
      if (raw == nullptr) {
        cout << "alloc failed\n";
        return -1;
      }

      cudaMemset(raw, -1, s);

      // check
      if (!checkCUDABytes(raw, -1, s)) {
        cout << "access failed\n";
        return -1;
      }

      bfcAlloc.Free(raw);
    }

    // check time for many allocation
    cout << "test many allocation\n";
#if 0
        std::vector<void*> ptrs;
        start = std::clock();
        for (int s = 1; s <= cnt; ++s) {
            void* raw = bfcAlloc.Alloc(s * cudaAlign);
            ptrs.push_back(raw);
        }
        duration = (std::clock() - start);
        cout << "alloc time: " << duration << '\n';


        start = std::clock();
        for (int s = 0; s < cnt; ++s) {
            bfcAlloc.Free(ptrs[s]);
        }
        duration = (std::clock() - start);
        cout << "free time: " << duration << '\n';
#endif
    start = std::clock();
    for (int s = 1; s <= cnt; ++s) {
      void* raw = bfcAlloc.Alloc(size);
      bfcAlloc.Free(raw);
    }
    duration = (std::clock() - start);
    cout << "alloc/free time: " << duration << '\n';

    std::cout << "test CUDA bfc arena allocator done\n";
  }
#endif

  return 1;
}
