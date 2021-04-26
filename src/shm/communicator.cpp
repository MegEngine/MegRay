/**
 * \file src/shm/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "communicator.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include <memory>
#include <mutex>
#include "megray/cuda_context.h"

namespace MegRay {

ShmCommunicator::ShmCommunicator(int ranks, int rank)
        : Communicator(ranks, rank) {
    alloc_cuda_stream();
}

void ShmCommunicator::free_shm() {
    CUDA_ASSERT(cudaHostUnregister(m_shmadd));
    int ret = shmdt(m_shmadd);
    shmctl(m_shmid, IPC_RMID, NULL);
    m_shmsize = 0;
}

Status ShmCommunicator::do_init() {
    return MEGRAY_OK;
}

Status ShmCommunicator::_recv(void* recvbuff, size_t size, uint32_t rank,
                              std::shared_ptr<Context> ctx) {
    // cuda stream synchronize
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    // call _irecv function
    MEGRAY_CHECK(_irecv(recvbuff, size, rank, ctx));
    return MEGRAY_OK;
}

Status ShmCommunicator::_send(const void* sendbuff, size_t size, uint32_t rank,
                              std::shared_ptr<Context> ctx) {
    // cuda stream synchronize
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    // call _isend function
    MEGRAY_CHECK(_isend(sendbuff, size, rank, ctx));
    return MEGRAY_OK;
}

Status ShmCommunicator::_irecv(void* recvbuff, size_t len, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(rank != m_rank, "invalid recv rank");
    MegRay::RingRecver recv;
    static MegRay::ChannelManager cm;
    // static int channel = 0;
    auto recv_cb = [=](raw_void_ptr shm_ptr) {
        auto stream = static_cast<CudaContext*>(ctx.get())->get_stream();
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, shm_ptr, len,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    MEGRAY_CHECK(recv.Update(rank, m_rank, cm.get_channel(rank, m_rank), len,
                             recv_cb));
    return MEGRAY_OK;
}

Status ShmCommunicator::_isend(const void* sendbuff, size_t len, uint32_t rank,
                               std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(rank != m_rank, "invalid recv rank");
    MegRay::RingSender send;
    static MegRay::ChannelManager cm;
    // static int channel = 0;
    auto send_cb = [=](raw_void_ptr shm_ptr) {
        auto stream = static_cast<CudaContext*>(ctx.get())->get_stream();
        CUDA_CHECK(cudaMemcpyAsync(shm_ptr, sendbuff, len,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    // the channel
    MEGRAY_CHECK(send.Notify(m_rank, rank, cm.get_channel(m_rank, rank), len,
                             send_cb));
    return MEGRAY_OK;
}

void* ShmCommunicator::alloc_shm(size_t size) {
    if (m_shmsize >= size) {
        return m_shmadd;
    }
    if (m_shmsize != 0)
        free_shm();
    m_shmsize = size;
    // todo random num
    m_shmkey = random();
    m_client->broadcast(&m_shmkey, &m_shmkey, sizeof(m_shmkey), 0);
    m_shmid = shmget(m_shmkey, size, IPC_CREAT | 0666);
    if (m_shmid < 0) {
        m_client->barrier();
        if (m_rank == 0) {
            m_shmid = shmget(m_shmkey, 1, IPC_CREAT | 0666);
            shmctl(m_shmid, IPC_RMID, NULL);
        }
        m_client->barrier();
        m_shmid = shmget(m_shmkey, size, IPC_CREAT | 0666);
    }
    MEGRAY_ASSERT(m_shmid >= 0, "shmget failed, error code %d", errno);
    m_shmadd = shmat(m_shmid, NULL, 0);
    CUDA_ASSERT(cudaHostRegister(m_shmadd, size, cudaHostRegisterMapped));
    m_client->barrier();
    return m_shmadd;
}

void ShmCommunicator::barrier_streams_local_process() {
    for (auto& stream : stream_vec) {
        CUDA_ASSERT(cudaStreamSynchronize(stream));
    }
}

void ShmCommunicator::alloc_cuda_stream() {
    if (stream_vec.size() >= m_nranks) {
        return;
    }
    for (int i = stream_vec.size(); i < m_nranks; i++) {
        cudaStream_t temp;
        CUDA_ASSERT(cudaStreamCreate(&temp));
        stream_vec.emplace_back(temp);
    }
}

void ShmCommunicator::free_cuda_stream() {
    for (auto& stream : stream_vec) {
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        CUDA_ASSERT(cudaStreamDestroy(stream));
    }
    stream_vec.clear();
}

ShmCommunicator::~ShmCommunicator() {
    if (stream_vec.size() > 0) {
        free_cuda_stream();
    }
    if (m_shmsize > 0)
        free_shm();
}
}  // namespace MegRay
