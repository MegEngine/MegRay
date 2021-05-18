/**
 * \file src/shm/utils.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
 #include "utils.h"

namespace MegRay {

template <typename T>
void cpu_reduce_kernel(T* dst, T* a, T* b, ReduceOp op, int len) {
    switch (op) {
        case MEGRAY_SUM:
            for (size_t i = 0; i < len; i++) {
                dst[i] = a[i] + b[i];
            }
            break;
        case MEGRAY_MAX:
            for (size_t i = 0; i < len; i++) {
                dst[i] = std::max(a[i], b[i]);
            }
            break;
        case MEGRAY_MIN:
            for (size_t i = 0; i < len; i++) {
                dst[i] = std::min(a[i], b[i]);
            }
            break;
        default:
            MEGRAY_THROW("unsupported op type");
    }
}

void cpu_reduce(void* dst, void* a, void* b, DType dtype, ReduceOp op,
                int len) {
    switch (dtype) {
        case MEGRAY_INT8:
            cpu_reduce_kernel<int8_t>((int8_t*)dst, (int8_t*)a, (int8_t*)b, op,
                                      len);
            break;
        case MEGRAY_UINT8:
            cpu_reduce_kernel<uint8_t>((uint8_t*)dst, (uint8_t*)a, (uint8_t*)b,
                                       op, len);
            break;
        case MEGRAY_INT32:
            cpu_reduce_kernel<int32_t>((int32_t*)dst, (int32_t*)a, (int32_t*)b,
                                       op, len);
            break;
        case MEGRAY_UINT32:
            cpu_reduce_kernel<uint32_t>((uint32_t*)dst, (uint32_t*)a,
                                        (uint32_t*)b, op, len);
            break;
        case MEGRAY_INT64:
            cpu_reduce_kernel<int64_t>((int64_t*)dst, (int64_t*)a, (int64_t*)b,
                                       op, len);
            break;
        case MEGRAY_FLOAT32:
            cpu_reduce_kernel<float>((float*)dst, (float*)a, (float*)b, op,
                                     len);
            break;
        default:
            MEGRAY_THROW("unsupported dtype");
    }
}

struct Param{
    volatile int* ptr;
    int x;
};

void CUDART_CB MyCallback_set(void* param){
    Param* p = (Param*)param;
    *p->ptr = p->x;
    delete p;
}

void CUDART_CB MyCallback_wait(void* param){
    Param* p = (Param*)param;
    while(*p->ptr < p->x);
    delete p;
}

void stream_set_signal(volatile int* ptr, int x, cudaStream_t stream) {
    auto param = new Param{ptr, x};
    CUDA_ASSERT(cudaLaunchHostFunc(stream, MyCallback_set, (void*)param));
}
void stream_wait_signal(volatile int* ptr, int x, cudaStream_t stream) {
    auto param = new Param{ptr, x};
    CUDA_ASSERT(cudaLaunchHostFunc(stream, MyCallback_wait, (void*)param));
}

}  // namespace MegRay