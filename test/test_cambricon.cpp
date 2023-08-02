#include <iostream>
#include <memory>
#include "megray.h"
#include "test_utils.h"

namespace MegRay {

#ifdef MEGRAY_WITH_CNCL

void* alloc_cambricon(size_t size) {
    void* result;
    CNRT_ASSERT(cnrtMalloc(&result, size));
    return result;
}
void set_device_cambricon(size_t device) {
    CNRT_ASSERT(cnrtSetDevice(device));
    std::cout << "set cnrt device" << device << std::endl;
}
void free_cambricon(void* ptr) {
    CNRT_ASSERT(cnrtFree(ptr));
}
std::shared_ptr<Context> make_context_cambricon() {
    cnrtQueue_t queue;
    CNRT_ASSERT(cnrtQueueCreate(&queue));
    auto context = std::make_shared<CnrtContext>(queue);
    return context;
}
void sync_context_cambricon(std::shared_ptr<Context> context) {
    MEGRAY_ASSERT(context->type() == MEGRAY_CTX_CNRT, "not a cnrt context");
    CNRT_ASSERT(cnrtQueueSync(
            static_cast<CnrtContext*>(context.get())->get_queue()));
}
void memcpy_h2d_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx) {
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    CNRT_ASSERT(cnrtMemcpyAsync(dst, src, len, queue, cnrtMemcpyHostToDev));
    CNRT_ASSERT(cnrtQueueSync(queue));
}
void memcpy_d2h_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx) {
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    CNRT_ASSERT(cnrtMemcpyAsync(dst, src, len, queue, cnrtMemcpyDevToHost));
    CNRT_ASSERT(cnrtQueueSync(queue));
}

#else

void* alloc_cambricon(size_t size) {
    return nullptr;
}
void set_device_cambricon(size_t device) {}
void free_cambricon(void* ptr) {}
std::shared_ptr<Context> make_context_cambricon() {
    return nullptr;
}
void sync_context_cambricon(std::shared_ptr<Context> context) {}
void memcpy_h2d_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx) {}
void memcpy_d2h_cambricon(void* dst, void* src, size_t len,
                          std::shared_ptr<Context> ctx) {}
#endif

}  // namespace MegRay