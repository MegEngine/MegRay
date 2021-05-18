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
#include "megray/debug.h"
#include "utils.h"

namespace MegRay {
// for allreduce only for now
void ShmCommunicator::persistent_thread() {
    while(1) {
        auto op = m_oplist.pop();
        if (op.is_nop) break;
        // reset shm signal
        op.send_signal[m_rank] = 0;
        op.reduce_signal[m_rank] = 0;
        _shm_barrier(op.shm_mutex);
        *(op.op_begin) = 1;

        // compute chunk sizes
        size_t size = get_dtype_size(op.dtype);
        size_t buff_size = op.len * size;
        size_t chunks = m_nranks * op.k;
        size_t quotient = op.len / chunks;
        size_t remainder = op.len % chunks;
        std::vector<size_t> chunk_sizes(chunks, quotient);
        for (size_t i = 0; i < remainder; i++) {
            chunk_sizes[i]++;
        }

        // compute chunk offsets
        std::vector<size_t> offsets(chunks, 0);
        for (size_t i = 1; i < chunks; i++) {
            offsets[i] = offsets[i - 1] + chunk_sizes[i - 1] * size;
        }
        size_t tmp_size = 0;
        for (size_t i = 0; i < op.k; i++) {
            tmp_size += chunk_sizes[m_rank * op.k + i];
        }

        size_t work_rank = ring_add(m_rank*op.k, op.k, chunks);
        for (size_t i = op.k+1;i <= chunks;i++) {
            while(op.send_signal[m_rank]<i);
            cpu_reduce((char*)op.shm_buffer + offsets[work_rank],
                       (char*)op.shm_buffer + offsets[work_rank],
                       (char*)op.shm_buffer + buff_size + offsets[work_rank],
                       op.dtype, op.op, chunk_sizes[work_rank]);
            if (i == chunks) _shm_barrier((volatile int*)op.shm_mutex);
            op.reduce_signal[m_rank] = i;
            work_rank = ring_add(work_rank, 1, chunks);
        }

        while(op.send_signal[m_rank] <= chunks);
        // kernal end
        _shm_barrier(op.shm_mutex);
    }
}

ShmCommunicator::ShmCommunicator(int ranks, int rank)
        : Communicator(ranks, rank) {
    alloc_cuda_stream();
    m_persistent_thread = std::thread(&ShmCommunicator::persistent_thread, this);
}

void ShmCommunicator::free_shm() {
    for (auto shm:m_shm_list) {
        CUDA_ASSERT(cudaHostUnregister(shm.addr));
        int ret = shmdt(shm.addr);
        shmctl(shm.shmid, IPC_RMID, NULL);
    }
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
    size_t max_size = 0;
    if (m_shm_list.size() > 0) {
        max_size = m_shm_list.back().size;
    }
    if (size <= max_size) return m_shm_list.back().addr;
    if (size > max_size && size < max_size*2) {
        size = max_size*2;
    }
    Shm tmp_shm;
    // todo random num
    tmp_shm.key = random();
    tmp_shm.size = size;
    m_client->broadcast(&tmp_shm.key, &tmp_shm.key, sizeof(tmp_shm.key), 0);
    tmp_shm.shmid = shmget(tmp_shm.key, size, IPC_CREAT | 0666);
    if (tmp_shm.shmid < 0) {
        m_client->barrier();
        if (m_rank == 0) {
            int shmid = shmget(tmp_shm.key, 1, IPC_CREAT | 0666);
            shmctl(shmid, IPC_RMID, NULL);
        }
        m_client->barrier();
        tmp_shm.shmid = shmget(tmp_shm.key, size, IPC_CREAT | 0666);
    }
    MEGRAY_ASSERT(tmp_shm.shmid >= 0, "shmget failed, error code %d", errno);
    tmp_shm.addr = shmat(tmp_shm.shmid, NULL, 0);
    CUDA_ASSERT(cudaHostRegister(tmp_shm.addr, size, cudaHostRegisterMapped));
    memset(tmp_shm.addr, 0, size);
    m_client->barrier();
    m_shm_list.push_back(tmp_shm);
    return tmp_shm.addr;
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
    Op op;
    op.is_nop = true;
    m_oplist.push(op);
    m_persistent_thread.join();
    free_shm();
}
}  // namespace MegRay
