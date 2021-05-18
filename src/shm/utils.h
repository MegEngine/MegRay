/**
 * \file src/shm/utils.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

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
