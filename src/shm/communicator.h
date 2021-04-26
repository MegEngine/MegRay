/**
 * \file src/shm/communicator.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include <memory>
#include <mutex>
#include <vector>
#include "megray/communicator.h"
#include "shm_ipc.h"

#ifdef MEGRAY_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif
namespace MegRay {
/*!
 * simple implementation of collective communications using shared memory
 * a shared memory communicator corresponds to a tcp communicator
 * */

class ShmCommunicator : public Communicator {
public:
    // constructor
    ShmCommunicator(int nranks, int rank);
    // deconstructor
    ~ShmCommunicator();
    // init function
    Status do_init() override;
    // _send function
    Status _send(const void* sendbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;
    // _recv function
    Status _recv(void* recvbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;

    // the scatter operation
    Status scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                   DType dtype, uint32_t root,
                   std::shared_ptr<Context> ctx) override;
    // the gather option
    Status gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                  DType dtype, uint32_t root,
                  std::shared_ptr<Context> ctx) override;
    // the all_to_all operation
    Status all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, std::shared_ptr<Context> ctx) override;
    // the all_gather operation
    Status all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                      DType dtype, std::shared_ptr<Context> ctx) override;
    // the all_reduce operation
    Status all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, ReduceOp op,
                      std::shared_ptr<Context> ctx) override;
    // the reduce scatter operation
    Status reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                          DType dtype, ReduceOp op,
                          std::shared_ptr<Context> ctx) override;
    // the broadcast operation
    Status broadcast(const void* sendbuff, void* recvbuff, size_t len,
                     DType dtype, uint32_t root,
                     std::shared_ptr<Context> ctx) override;
    // the reduce operation
    Status reduce(const void* sendbuff, void* recvbuff, size_t len, DType dtype,
                  ReduceOp op, uint32_t root,
                  std::shared_ptr<Context> ctx) override;

protected:
    // allocate cuda stream for parllel
    void alloc_cuda_stream();
    // destroy parllel stream
    void free_cuda_stream();
    // barrier current streams
    void barrier_streams_local_process();
    // use to free shared memory
    void free_shm();
    // alloc shared memory with given size
    void* alloc_shm(size_t size);

    // using shm broadcast data
    Status _shm_broadcast(const void* sendbuff, void* recvbuff, size_t len,
                          DType dtype, uint32_t root,
                          std::shared_ptr<Context> ctx);
    // using shm all reduce
    Status _shm_all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                           DType dtype, ReduceOp op,
                           std::shared_ptr<Context> ctx);
    // using shm all gather algorithm
    Status _shm_all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                           DType dtype, std::shared_ptr<Context> ctx);
    // using shm all to all algorithm
    Status _shm_all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                           DType dtype, std::shared_ptr<Context> ctx);
    // using shm gather algorithm
    Status _shm_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                       DType dtype, uint32_t root,
                       std::shared_ptr<Context> ctx);
    // using shm reduce algorithm
    Status _shm_reduce(const void* sendbuff, void* recvbuff, size_t len,
                       DType dtype, ReduceOp op, uint32_t root,
                       std::shared_ptr<Context> ctx);
    // using shm to do reduce scatter
    Status _shm_reduce_scatter(const void* sendbuff, void* recvbuff,
                               size_t recvlen, DType dtype, ReduceOp op,
                               std::shared_ptr<Context> ctx);
    // using shm to recv data
    Status _irecv(void* recvbuff, size_t len, uint32_t rank,
                  std::shared_ptr<Context> ctx);
    // using shm to send data
    Status _isend(const void* sendbuff, size_t len, uint32_t rank,
                  std::shared_ptr<Context> ctx);
    // used for barrier the sum
    void _shm_barrier_sum(volatile int* mutex);
    // used for barrier the all reduce
    void _shm_barrier(volatile int* mutex);

private:
    // parllel mutex
    std::mutex m_requests_mtx;
    // use for parllell in device
    std::vector<cudaStream_t> stream_vec;
    // the shm pointer which will be registered in cudastream
    void* m_shmadd;
    // shm size
    size_t m_shmsize = 0;
    // shmid for the process
    int m_shmid;
    // m_shmkey save for shm
    key_t m_shmkey;
    // single flag that label in single machine
    bool m_is_single_machine = true;
};

}  // namespace MegRay

