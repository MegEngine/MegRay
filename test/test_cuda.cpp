/**
 * \file test/test_cuda.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megray.h"

#include "test_utils.h"

namespace MegRay {

#ifdef MEGRAY_WITH_CUDA

void* alloc_cuda(size_t size) {
    void* result;
    CUDA_ASSERT(cudaMalloc(&result, size));
    return result;
}

void free_cuda(void* ptr) {
    CUDA_ASSERT(cudaFree(ptr));
}

void set_device_cuda(size_t device) {
    CUDA_ASSERT(cudaSetDevice(device));
}

std::shared_ptr<Context> make_context_cuda() {
    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream));
    auto context = std::make_shared<CudaContext>(stream);
    return context;
}

void sync_context_cuda(std::shared_ptr<Context> context) {
    MEGRAY_ASSERT(context->type() == MEGRAY_CTX_CUDA, "not a cuda context");
    CUDA_ASSERT(cudaStreamSynchronize(
            static_cast<CudaContext*>(context.get())->get_stream()));
}

void memcpy_h2d_cuda(void* dst, void* src, size_t len) {
    CUDA_ASSERT(cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice));
}

void memcpy_d2h_cuda(void* dst, void* src, size_t len) {
    CUDA_ASSERT(cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost));
}

#else  // MEGRAY_WITH_CUDA

void* alloc_cuda(size_t size) {
    return nullptr;
}
void set_device_cuda(size_t device) {}
void free_cuda(void* ptr) {}
std::shared_ptr<Context> make_context_cuda() {
    return nullptr;
}
void sync_context_cuda(std::shared_ptr<Context> context) {}
void memcpy_h2d_cuda(void* dst, void* src, size_t len) {}
void memcpy_d2h_cuda(void* dst, void* src, size_t len) {}

#endif  // MEGRAY_WITH_CUDA

}  // namespace MegRay
