#pragma once

#include <cstdint>
#include "cuda_runtime.h"
#include "megray/common.h"

namespace MegRay {

inline uint32_t ring_add(uint32_t n, uint32_t delta, uint32_t m) {
    return (n + delta) % m;
}

inline uint32_t ring_sub(uint32_t n, uint32_t delta, uint32_t m) {
    return (n + m - delta % m) % m;
}

void stream_set_signal(volatile int* ptr, int x, cudaStream_t stream);
void stream_wait_signal(volatile int* ptr, int x, cudaStream_t stream);

void cpu_reduce(void* dst, void* a, void* b, DType dtype, ReduceOp op, int len);
}  // namespace MegRay
