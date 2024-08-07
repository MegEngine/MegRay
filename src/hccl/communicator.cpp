#include "communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "megray/aclrt_context.h"
#include "megray/common.h"
#include "utils.h"

namespace MegRay {

class HcclCommunicatorPrivate {
public:
    HcclComm m_comm;
    ~HcclCommunicatorPrivate() { HcclCommDestroy(m_comm); }
};

HcclCommunicator::HcclCommunicator(int nranks, int rank)
        : Communicator(nranks, rank) {}

HcclCommunicator::~HcclCommunicator() {}

Status HcclCommunicator::do_init(BcastCallback cb) {
    int device_id = -1;
    ACL_CHECK(aclrtGetDevice(&device_id));
    if (device_id == -1) {
        ACL_CHECK(aclrtSetDevice(m_rank));
    } else {
        MEGRAY_ASSERT(device_id == int(m_rank), "device_id: %d, rank: %u",
                      device_id, m_rank);
    }

    HcclRootInfo root_info;
    uint32_t root = 0;
    if (m_rank == root) {
        MEGRAY_HCCL_ASSERT(HcclGetRootInfo(&root_info));
    }

    if (cb) {
        cb(reinterpret_cast<char*>(&root_info), HCCL_ROOT_INFO_BYTES);
    } else {
        MEGRAY_CHECK(m_client->broadcast(&root_info, &root_info,
                                         HCCL_ROOT_INFO_BYTES, root));
    }

    m_hccl = std::make_unique<HcclCommunicatorPrivate>();
    MEGRAY_HCCL_ASSERT(HcclCommInitRootInfo(m_nranks, &root_info, m_rank,
                                            &m_hccl->m_comm));
    return MEGRAY_OK;
}

Status HcclCommunicator::do_init() {
    return do_init({});
}

Status HcclCommunicator::_send(const void* sendbuff, size_t size, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclSend(const_cast<void*>(sendbuff), size,
                                HCCL_DATA_TYPE_UINT8, rank, m_hccl->m_comm,
                                stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::_recv(void* recvbuff, size_t size, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclRecv(recvbuff, size, HCCL_DATA_TYPE_UINT8, rank,
                                m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::scatter(const void* sendbuff, void* recvbuff,
                                 size_t recvlen, DType dtype, uint32_t root,
                                 std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclScatter(const_cast<void*>(sendbuff), recvbuff,
                                   recvlen, as_hccl_dtype(dtype), root,
                                   m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::gather(const void*, void*, size_t, DType, uint32_t,
                                std::shared_ptr<Context>) {
    MEGRAY_ERROR("megray: the hccl backend does not support gather");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status HcclCommunicator::all_to_all(const void* sendbuff, void* recvbuff,
                                    size_t len, DType dtype,
                                    std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclAlltoAll(sendbuff, len, as_hccl_dtype(dtype),
                                    recvbuff, len, as_hccl_dtype(dtype),
                                    m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::all_gather(const void* sendbuff, void* recvbuff,
                                    size_t sendlen, DType dtype,
                                    std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclAllGather(const_cast<void*>(sendbuff), recvbuff,
                                     sendlen, as_hccl_dtype(dtype),
                                     m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::all_reduce(const void* sendbuff, void* recvbuff,
                                    size_t len, DType dtype, ReduceOp op,
                                    std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclAllReduce(
            const_cast<void*>(sendbuff), recvbuff, len, as_hccl_dtype(dtype),
            as_hccl_reduce_op(op), m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff,
                                        size_t recvlen, DType dtype,
                                        ReduceOp op,
                                        std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();
    MEGRAY_HCCL_ASSERT(HcclReduceScatter(const_cast<void*>(sendbuff), recvbuff,
                                         recvlen, as_hccl_dtype(dtype),
                                         as_hccl_reduce_op(op), m_hccl->m_comm,
                                         stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::broadcast(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype, uint32_t root,
                                   std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();

    if (m_rank == root) {
        size_t elem_size = get_dtype_size(dtype);
        if (reinterpret_cast<uintptr_t>(recvbuff) % 64 != 0 ||
            reinterpret_cast<uintptr_t>(const_cast<void*>(sendbuff)) % 64 !=
                    0) {
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(recvbuff, len * elem_size, const_cast<void*>(sendbuff),
                        len * elem_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        } else {
            aclrtMemcpyAsync(recvbuff, len * elem_size,
                             const_cast<void*>(sendbuff), len * elem_size,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        MEGRAY_HCCL_ASSERT(HcclBroadcast(const_cast<void*>(sendbuff), len,
                                         as_hccl_dtype(dtype), root,
                                         m_hccl->m_comm, stream));
    } else {
        MEGRAY_HCCL_ASSERT(HcclBroadcast(recvbuff, len, as_hccl_dtype(dtype),
                                         root, m_hccl->m_comm, stream));
    }
    return MEGRAY_OK;
}

Status HcclCommunicator::reduce(const void* sendbuff, void* recvbuff,
                                size_t len, DType dtype, ReduceOp op,
                                uint32_t root, std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_ACLRT,
                  "context type must be acl, but got %d", ctx->type());
    auto stream = std::static_pointer_cast<AclrtContext>(ctx)->get_stream();

    MEGRAY_HCCL_ASSERT(HcclReduce(const_cast<void*>(sendbuff), recvbuff, len,
                                  as_hccl_dtype(dtype), as_hccl_reduce_op(op),
                                  root, m_hccl->m_comm, stream));
    return MEGRAY_OK;
}

Status HcclCommunicator::group_start() {
    MEGRAY_ERROR("megray: the hccl backend does not support group_start");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status HcclCommunicator::group_end() {
    MEGRAY_ERROR("megray: the hccl backend does not support group_end");
    return MEGRAY_NOT_IMPLEMENTED;
}

}  // namespace MegRay
