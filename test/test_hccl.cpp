#include <memory>
#include "megray.h"
#include "test_utils.h"

namespace MegRay {

#ifdef MEGRAY_WITH_HCCL

void* alloc_ascend(size_t size) {
    void* result;
    ACL_ASSERT(aclrtMalloc(&result, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    return result;
}

void set_device_ascend(size_t device) {
    ACL_ASSERT(aclrtSetDevice(device));
}

void free_ascend(void* ptr) {
    ACL_ASSERT(aclrtFree(ptr));
}

std::shared_ptr<Context> make_context_ascend() {
    aclrtStream stream;
    ACL_ASSERT(aclrtCreateStream(&stream));
    auto context = std::make_shared<AclrtContext>(stream);
    return context;
}

void sync_context_ascend(std::shared_ptr<Context> context) {
    MEGRAY_ASSERT(context->type() == MEGRAY_CTX_ACLRT, "not a acl context");
    ACL_ASSERT(aclrtSynchronizeStream(
            static_cast<AclrtContext*>(context.get())->get_stream()));
}

void memcpy_h2d_ascend(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> ctx) {
    auto stream = static_cast<AclrtContext*>(ctx.get())->get_stream();
    ACL_ASSERT(aclrtMemcpyAsync(dst, len, src, len, ACL_MEMCPY_HOST_TO_DEVICE,
                                stream));
    ACL_ASSERT(aclrtSynchronizeStream(stream));
}

void memcpy_d2h_ascend(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> ctx) {
    auto stream = static_cast<AclrtContext*>(ctx.get())->get_stream();
    ACL_ASSERT(aclrtMemcpyAsync(dst, len, src, len, ACL_MEMCPY_DEVICE_TO_HOST,
                                stream));
    ACL_ASSERT(aclrtSynchronizeStream(stream));
}

#else

void* alloc_ascend(size_t size) {
    return nullptr;
}

void set_device_ascend(size_t device) {}

void free_ascend(void* ptr) {}

std::shared_ptr<Context> make_context_ascend() {
    return nullptr;
}

void sync_context_ascend(std::shared_ptr<Context> context) {}

void memcpy_h2d_ascend(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> ctx) {}

void memcpy_d2h_ascend(void* dst, void* src, size_t len,
                       std::shared_ptr<Context> ctx) {}

#endif

}  // namespace MegRay
