#pragma once

#include <cstdint>
#include "cuda_runtime.h"

namespace MegRay {

// launch cuda kernel for reduce operations
void reduce(void* i0, void* i1, void* o, size_t len, DType dtype, ReduceOp op,
            cudaStream_t stream);

inline uint32_t ring_add(uint32_t n, uint32_t delta, uint32_t m) {
    return (n + delta) % m;
}

inline uint32_t ring_sub(uint32_t n, uint32_t delta, uint32_t m) {
    return (n + m - delta % m) % m;
}

}  // namespace MegRay
