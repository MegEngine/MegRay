#include "communicator.h"
#include <cncl.h>
#include <cnrt.h>
#include "megray/cnrt_context.h"
#include "megray/common.h"
#include "utils.h"

namespace MegRay {

class CnclCommunicatorPrivate {
public:
    cnclComm_t m_comm;
    ~CnclCommunicatorPrivate() { cnclDestroyComms(&m_comm, 1); }
};

CnclCommunicator::CnclCommunicator(int nranks, int rank)
        : Communicator(nranks, rank) {}

CnclCommunicator::~CnclCommunicator() {}

Status CnclCommunicator::do_init() {
    uint32_t root = 0;
    cnclCliqueId uid;
    if (m_rank == root) {
        MEGRAY_CNCL_ASSERT(cnclGetCliqueId(&uid));
    }
    MEGRAY_CHECK(
            m_client->broadcast(&uid, &uid, CNCL_CLIQUE_ID_BYTES_SIZE, root));
    m_cncl = std::make_unique<CnclCommunicatorPrivate>();
    int device_id;
    CNRT_CHECK(cnrtGetDevice(&device_id));
    auto u = reinterpret_cast<size_t*>(&uid);
    int dev_list[1];
    int rank_list[1];
    dev_list[0] = device_id;
    rank_list[0] = m_rank;
    MEGRAY_CNCL_ASSERT(cnclInitComms(&m_cncl->m_comm, 1, dev_list, rank_list,
                                     m_nranks, &uid));
    return MEGRAY_OK;
}

Status CnclCommunicator::do_init(BcastCallback cb) {
    uint32_t root = 0;
    cnclCliqueId_t uid;
    if (m_rank == root) {
        cnclGetCliqueId(uid);
    }
    cb(uid->data, CNCL_CLIQUE_ID_BYTES_SIZE);
    m_cncl = std::make_unique<CnclCommunicatorPrivate>();
    int device_id;
    CNRT_CHECK(cnrtGetDevice(&device_id));
    MEGRAY_CNCL_CHECK(cnclInitComms(&m_cncl->m_comm, 1, &device_id,
                                    (int*)&m_rank, m_nranks, uid));
    return MEGRAY_OK;
}

Status CnclCommunicator::_send(const void* sendbuff, size_t size, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    // perform cncl send synchronously
    MEGRAY_CNCL_CHECK(cnclSend(const_cast<void*>(sendbuff), size, cnclChar,
                               rank, m_cncl->m_comm, queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::_recv(void* recvbuff, size_t size, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    // perform cncl send synchronously
    MEGRAY_CNCL_CHECK(
            cnclRecv(recvbuff, size, cnclChar, rank, m_cncl->m_comm, queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::scatter(const void* sendbuff, void* recvbuff,
                                 size_t recvlen, DType dtype, uint32_t root,
                                 std::shared_ptr<Context> ctx) {
    MEGRAY_ERROR("megray: not impl scatter");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status CnclCommunicator::gather(const void* sendbuff, void* recvbuff,
                                size_t sendlen, DType dtype, uint32_t root,
                                std::shared_ptr<Context> ctx) {
    MEGRAY_ERROR("megray: not impl gather");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status CnclCommunicator::all_to_all(const void* sendbuff, void* recvbuff,
                                    size_t len, DType dtype,
                                    std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    cnclDataType_t cncl_dtype = get_cncl_dtype(dtype);
    MEGRAY_CNCL_CHECK(cnclAlltoAll(sendbuff, recvbuff, len, cncl_dtype,
                                   m_cncl->m_comm, queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::all_gather(const void* sendbuff, void* recvbuff,
                                    size_t sendlen, DType dtype,
                                    std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();

    MEGRAY_CNCL_CHECK(cnclAllGather(sendbuff, recvbuff, sendlen,
                                    get_cncl_dtype(dtype), m_cncl->m_comm,
                                    queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::all_reduce(const void* sendbuff, void* recvbuff,
                                    size_t len, DType dtype, ReduceOp op,
                                    std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();

    MEGRAY_CNCL_CHECK(
            cnclAllReduce(sendbuff, recvbuff, len, get_cncl_dtype(dtype),
                          get_cncl_reduce_op(op), m_cncl->m_comm, queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff,
                                        size_t recvlen, DType dtype,
                                        ReduceOp op,
                                        std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    // perform reduce scatter synchronously
    MEGRAY_CNCL_CHECK(cnclReduceScatter(
            sendbuff, recvbuff, recvlen, get_cncl_dtype(dtype),
            get_cncl_reduce_op(op), m_cncl->m_comm, queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::broadcast(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype, uint32_t root,
                                   std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    // perform reduce scatter synchronously
    MEGRAY_CNCL_CHECK(cnclBroadcast(sendbuff, recvbuff, len,
                                    get_cncl_dtype(dtype), root, m_cncl->m_comm,
                                    queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::reduce(const void* sendbuff, void* recvbuff,
                                size_t len, DType dtype, ReduceOp op,
                                uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get cnrt queue
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CNRT,
                  "only cnrt context supported");
    cnrtQueue_t queue = static_cast<CnrtContext*>(ctx.get())->get_queue();
    // perform reduce scatter synchronously
    MEGRAY_CNCL_CHECK(cnclReduce(sendbuff, recvbuff, len, get_cncl_dtype(dtype),
                                 get_cncl_reduce_op(op), root, m_cncl->m_comm,
                                 queue));
    return MEGRAY_OK;
}

Status CnclCommunicator::group_start() {
    MEGRAY_ERROR("megray: not impl group start");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status CnclCommunicator::group_end() {
    MEGRAY_ERROR("megray: not impl group end");
    return MEGRAY_NOT_IMPLEMENTED;
}

}  // namespace MegRay