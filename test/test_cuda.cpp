#include <memory>
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

void memcpy_h2d_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> ctx) {
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_ASSERT(cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));
}

void memcpy_d2h_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> ctx) {
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_ASSERT(cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));
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
void memcpy_h2d_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> ctx) {}
void memcpy_d2h_cuda(void* dst, void* src, size_t len,
                     std::shared_ptr<Context> ctx) {}

#endif  // MEGRAY_WITH_CUDA

}  // namespace MegRay
