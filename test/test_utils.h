#pragma once

#include <memory>

#include "megray/core/context.h"

namespace MegRay{

struct ContextTrait{
    void* (*alloc)(size_t size);
    void (*set_device)(size_t device);
    void (*free)(void* ptr);
    std::shared_ptr<Context> (*make_context)();
    void (*sync_context)(std::shared_ptr<Context> context);
    void (*memcpy_h2d)(void* dst, void* src, size_t len);
    void (*memcpy_d2h)(void* dst, void* src, size_t len);
};

void* alloc_cuda(size_t size);
void set_device_cuda(size_t device);
void free_cuda(void* ptr);
std::shared_ptr<Context> make_context_cuda();
void sync_context_cuda(std::shared_ptr<Context> context);
void memcpy_h2d_cuda(void* dst, void* src, size_t len);
void memcpy_d2h_cuda(void* dst, void* src, size_t len);

void* alloc_hip(size_t size);
void set_device_hip(size_t device);
void free_hip(void* ptr);
std::shared_ptr<Context> make_context_hip();
void sync_context_hip(std::shared_ptr<Context> context);
void memcpy_h2d_hip(void* dst, void* src, size_t len);
void memcpy_d2h_hip(void* dst, void* src, size_t len);

static ContextTrait context_trait_array[MEGRAY_CTX_COUNT] = {
    {},
    {&alloc_cuda, &set_device_cuda, &free_cuda, &make_context_cuda, &sync_context_cuda, &memcpy_h2d_cuda, &memcpy_d2h_cuda},
    {&alloc_hip, &set_device_hip, &free_hip, &make_context_hip, &sync_context_hip, &memcpy_h2d_hip, &memcpy_d2h_hip}
};

static ContextType get_preferred_context(Backend backend){
    switch(backend){
        case MEGRAY_NCCL:
            return MEGRAY_CTX_CUDA;
        case MEGRAY_RCCL:
            return MEGRAY_CTX_HIP;
        case MEGRAY_UCX:
            return MEGRAY_CTX_CUDA;
        default:
            return MEGRAY_CTX_DEFAULT;
    }
}

static ContextTrait get_context_trait(ContextType type){
    return context_trait_array[type];
}

}