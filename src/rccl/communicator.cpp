/**
 * \file src/rccl/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "communicator.h"

#include "nccl.h"

#include "utils.h"
#include "megray/hip_context.h"

#define CHECK_LAUNCH_MODE                                                  \
    do {                                                                   \
        const char* str = getenv("RCCL_LAUNCH_MODE");                      \
        if (!str or strcmp(str, "PARALLEL") != 0) {                        \
            MEGRAY_ERROR("please set RCCL_LAUNCH_MODE to \"PARALLEL\"\n"); \
            return MEGRAY_ENV_ERROR;                                       \
        }                                                                  \
    } while (0)

namespace MegRay {

class RcclCommunicatorPrivate{
public:
    ncclComm_t m_comm;
    ~RcclCommunicatorPrivate(){
        ncclCommDestroy(m_comm);
    }
};

RcclCommunicator::RcclCommunicator(int nranks, int rank):
        Communicator(nranks, rank) {
}

RcclCommunicator::~RcclCommunicator(){}

Status RcclCommunicator::do_init() {
    uint32_t root = 0;
    ncclUniqueId uid;
    if (m_rank == root) {
        ncclGetUniqueId(&uid);
    }
    MEGRAY_CHECK(m_client->broadcast(&uid, &uid, NCCL_UNIQUE_ID_BYTES, root));
    m_rccl = std::make_unique<RcclCommunicatorPrivate>();
    RCCL_CHECK(ncclCommInitRank(&m_rccl->m_comm, m_nranks, uid, m_rank));
    return MEGRAY_OK;
}

Status RcclCommunicator::send(const void* sendbuff, size_t len, uint32_t rank,
        std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform nccl send synchronously
    RCCL_CHECK(ncclSend(sendbuff, len, ncclChar, rank, m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::recv(void* recvbuff, size_t len, uint32_t rank,
        std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform nccl send synchronously
    RCCL_CHECK(ncclRecv(recvbuff, len, ncclChar, rank, m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::scatter(const void* sendbuff, void* recvbuff,
        size_t recvlen, DType dtype, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
    CHECK_LAUNCH_MODE;
    // perform nccl send/recv in a group
    ncclGroupStart();
    if (m_rank == root) {
        for (size_t r = 0; r < m_nranks; r++) {
            const char* p = (const char*)sendbuff + r * recvlen * get_dtype_size(dtype);
            RCCL_CHECK(ncclSend((const void*)p, recvlen, nccl_dtype, r, m_rccl->m_comm, stream));
        }
    }
    RCCL_CHECK(ncclRecv(recvbuff, recvlen, nccl_dtype, root, m_rccl->m_comm, stream));
    ncclGroupEnd();
    // hip stream synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::gather(const void* sendbuff, void* recvbuff,
        size_t sendlen, DType dtype, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
    CHECK_LAUNCH_MODE;
    // perform nccl send/recv in a group
    ncclGroupStart();
    if (m_rank == root) {
        for (size_t r = 0; r < m_nranks; r++) {
            char* p = (char*)recvbuff + r * sendlen * get_dtype_size(dtype);
            RCCL_CHECK(ncclRecv((void*)p, sendlen, nccl_dtype, r, m_rccl->m_comm, stream));
        }
    }
    RCCL_CHECK(ncclSend(sendbuff, sendlen, nccl_dtype, root, m_rccl->m_comm, stream));
    ncclGroupEnd();
    // hip stream synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::all_to_all(const void* sendbuff, void* recvbuff,
        size_t len, DType dtype, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
    CHECK_LAUNCH_MODE;
    // perform nccl send/recv in a group
    ncclGroupStart();
    for (size_t r = 0; r < m_nranks; r++) {
        const char* p = (const char*)sendbuff + r * len * get_dtype_size(dtype);
        char* q = (char*)recvbuff + r * len * get_dtype_size(dtype);
        RCCL_CHECK(ncclSend((const void*)p, len, nccl_dtype, r, m_rccl->m_comm, stream));
        RCCL_CHECK(ncclRecv((void*)q, len, nccl_dtype, r, m_rccl->m_comm, stream));
    }
    ncclGroupEnd();
    // hip stream synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
        DType dtype, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform all gather synchronously
    RCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendlen, get_nccl_dtype(dtype),
            m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::all_reduce(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform all reduce synchronously
    RCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
        DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform reduce scatter synchronously
    RCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvlen, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::broadcast(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform broadcast synchronously
    RCCL_CHECK(ncclBroadcast(sendbuff, recvbuff, len, get_nccl_dtype(dtype), root,
            m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status RcclCommunicator::reduce(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, ReduceOp op, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get hip stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_HIP, "only hip context supported");
    hipStream_t stream = static_cast<HipContext*>(ctx.get())->get_stream();
    // perform reduce synchronously
    RCCL_CHECK(ncclReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), root, m_rccl->m_comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return MEGRAY_OK;
}

} // namespace MegRay
