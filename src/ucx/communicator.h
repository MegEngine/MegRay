/**
 * \file src/ucx/communicator.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <mutex>
#include <vector>

#include <ucp/api/ucp.h>

#include "../communicator.h"

namespace MegRay {

/*!
 * simple implementation of collective communications using ucp api
 * a ucx communicator corresponds to a ucp worker
 */
class UcxCommunicator : public Communicator {
    public:
        UcxCommunicator(int nranks, int rank);

        ~UcxCommunicator();

        // get a serialized string of ucp worker address
        std::string get_uid() override;

        Status init(const std::vector<std::string>& uids) override;

        Status send(const void* sendbuff, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) override;

        Status recv(void* recvbuff, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) override;

        Status all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                DType dtype, std::shared_ptr<Context> ctx) override;

        Status all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) override;

        Status reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) override;

        Status broadcast(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) override;

        Status reduce(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, ReduceOp op, uint32_t root, std::shared_ptr<Context> ctx) override;

    private:
        // internal non-blocking send method
        Status _send(const void* sendbuff, size_t len, uint32_t rank);

        // internal non-blocking receive method
        Status _recv(void* recvbuff, size_t len, uint32_t rank);

        // flush _send and _recv requests
        Status _flush();

        // launch cuda kernel for reduce operations
        void _reduce(void* i0, void* i1, void* o, size_t len, DType dtype,
                ReduceOp op, cudaStream_t stream);

        ucp_context_h m_context;
        ucp_worker_h m_worker;
        bool m_inited;
        std::vector<ucp_ep_h> m_eps;  // ucp endpoints
        std::vector<void*> m_requests;
        std::mutex m_requests_mtx;
};

} // namespace MegRay
