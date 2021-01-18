/**
 * \file test/test_hip.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megray.h"

#include "test_utils.h"

#include <cassert>

namespace MegRay {

#ifdef MEGRAY_WITH_HIP

void* alloc_hip(size_t size) {
    void* result;
    HIP_ASSERT(hipMalloc(&result, size));
    return result;
}

void free_hip(void* ptr) {
    HIP_ASSERT(hipFree(ptr));
}

void set_device_hip(size_t device) {
    HIP_ASSERT(hipSetDevice(device));
}

std::shared_ptr<Context> make_context_hip() {
    hipStream_t stream;
    HIP_ASSERT(hipStreamCreate(&stream));
    auto context = std::make_shared<HipContext>(stream);
    return context;
}

void sync_context_hip(std::shared_ptr<Context> context) {
    MEGRAY_ASSERT(context->type() == MEGRAY_CTX_HIP, "not a hip context");
    HIP_ASSERT(hipStreamSynchronize(
            static_cast<HipContext*>(context.get())->get_stream()));
}

void memcpy_h2d_hip(void* dst, void* src, size_t len) {
    HIP_ASSERT(hipMemcpy(dst, src, len, hipMemcpyHostToDevice));
}

void memcpy_d2h_hip(void* dst, void* src, size_t len) {
    HIP_ASSERT(hipMemcpy(dst, src, len, hipMemcpyDeviceToHost));
}

#else  // MEGRAY_WITH_HIP

void* alloc_hip(size_t size) {
    return nullptr;
}
void set_device_hip(size_t device) {}
void free_hip(void* ptr) {}
std::shared_ptr<Context> make_context_hip() {
    return nullptr;
}
void sync_context_hip(std::shared_ptr<Context> context) {}
void memcpy_h2d_hip(void* dst, void* src, size_t len) {}
void memcpy_d2h_hip(void* dst, void* src, size_t len) {}

#endif  // MEGRAY_WITH_HIP

}  // namespace MegRay
