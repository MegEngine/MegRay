/**
 * \file src/nccl/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "communicator.h"

#include <string.h>

#include "utils.h"

namespace MegRay {

NcclCommunicator::NcclCommunicator(int nranks, int rank) :
        Communicator(nranks, rank), m_inited(false) {
    NCCL_ASSERT(ncclGetUniqueId(&m_uid));
}

NcclCommunicator::~NcclCommunicator() {
    if (m_inited) {
        ncclCommDestroy(m_comm);
    }
}

std::string NcclCommunicator::get_uid() {
    // serialize ncclUniqueId into a string
    return std::string(m_uid.internal, NCCL_UNIQUE_ID_BYTES);
}

Status NcclCommunicator::init(const std::vector<std::string>& uids) {
    MEGRAY_ASSERT(uids.size() == m_nranks, "incorrect size of uids");
    // only use unique id of rank 0 for initialization
    const std::string uid = uids[0];
    MEGRAY_ASSERT(uid.size() == NCCL_UNIQUE_ID_BYTES, "invalid uid");
    memcpy(m_uid.internal, uid.data(), NCCL_UNIQUE_ID_BYTES);
    // initialize nccl communicator
    NCCL_CHECK(ncclCommInitRank(&m_comm, m_nranks, m_uid, m_rank));
    m_inited = true;
    return MEGRAY_OK;
}

Status NcclCommunicator::send(const void* sendbuff, size_t len, uint32_t rank,
        std::shared_ptr<Context> ctx) {
    // derived from base class, not implemented
    MEGRAY_THROW("not implemented");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status NcclCommunicator::recv(void* recvbuf, size_t len, uint32_t rank,
        std::shared_ptr<Context> ctx) {
    // derived from base class, not implemented
    MEGRAY_THROW("not implemented");
    return MEGRAY_NOT_IMPLEMENTED;
}

Status NcclCommunicator::all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
        DType dtype, std::shared_ptr<Context> ctx) {
    // check context type and get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // perform all gather synchronously
    NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendlen, get_nccl_dtype(dtype),
            m_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status NcclCommunicator::all_reduce(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) {
    // check context type and get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // perform all reduce synchronously
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), m_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status NcclCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
        DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) {
    // check context type and get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // perform reduce scatter synchronously
    NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvlen, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), m_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status NcclCommunicator::broadcast(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // perform broadcast synchronously
    NCCL_CHECK(ncclBroadcast(sendbuff, recvbuff, len, get_nccl_dtype(dtype), root,
            m_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status NcclCommunicator::reduce(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, ReduceOp op, uint32_t root, std::shared_ptr<Context> ctx) {
    // check context type and get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // perform reduce synchronously
    NCCL_CHECK(ncclReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
            get_nccl_reduce_op(op), root, m_comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

} // namespace MegRay
