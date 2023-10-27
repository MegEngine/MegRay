#pragma once

#include "megray.h"

namespace MegRay {

struct ContextTrait {
    void* (*alloc)(size_t size);
    void (*set_device)(size_t device);
    void (*free)(void* ptr);
    std::shared_ptr<Context> (*make_context)();
    void (*sync_context)(std::shared_ptr<Context> context);
    void (*memcpy_h2d)(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> context);
    void (*memcpy_d2h)(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> context);
};

void* alloc_cuda(size_t size);
void set_device_cuda(size_t device);
void free_cuda(void* ptr);
std::shared_ptr<Context> make_context_cuda();
void sync_context_cuda(std::shared_ptr<Context> context);
void memcpy_h2d_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> context);
void memcpy_d2h_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> context);

void* alloc_hip(size_t size);
void set_device_hip(size_t device);
void free_hip(void* ptr);
std::shared_ptr<Context> make_context_hip();
void sync_context_hip(std::shared_ptr<Context> context);
void memcpy_h2d_hip(void* dst, void* src, size_t len,
                    std::shared_ptr<Context> ctx);
void memcpy_d2h_hip(void* dst, void* src, size_t len,
                    std::shared_ptr<Context> ctx);

void* alloc_cambricon(size_t size);
void set_device_cambricon(size_t device);
void free_cambricon(void* ptr);
std::shared_ptr<Context> make_context_cambricon();
void sync_context_cambricon(std::shared_ptr<Context> context);
void memcpy_h2d_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx);
void memcpy_d2h_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx);

static ContextTrait context_trait_array[MEGRAY_CTX_COUNT] = {
        {},
        {&alloc_cuda, &set_device_cuda, &free_cuda, &make_context_cuda,
         &sync_context_cuda, &memcpy_h2d_cuda, &memcpy_d2h_cuda},
        {&alloc_hip, &set_device_hip, &free_hip, &make_context_hip,
         &sync_context_hip, &memcpy_h2d_hip, &memcpy_d2h_hip},
        {&alloc_cambricon, &set_device_cambricon, &free_cambricon,
         &make_context_cambricon, &sync_context_cambricon,
         &memcpy_h2d_cambricon, &memcpy_d2h_cambricon},
};

static ContextType get_preferred_context(Backend backend) {
    switch (backend) {
        case MEGRAY_NCCL:
            return MEGRAY_CTX_CUDA;
        case MEGRAY_RCCL:
            return MEGRAY_CTX_HIP;
        case MEGRAY_UCX:
            return MEGRAY_CTX_CUDA;
        case MEGRAY_SHM:
            return MEGRAY_CTX_CUDA;
        case MEGRAY_CNCL:
            return MEGRAY_CTX_CNRT;
        default:
            return MEGRAY_CTX_DEFAULT;
    }
}

static ContextTrait get_context_trait(ContextType type) {
    return context_trait_array[type];
}

}  // namespace MegRay
