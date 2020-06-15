/**
 * \file src/nccl/communicator.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <string>

#include "nccl.h"

#include "../communicator.h"

namespace MegRay {

/*!
 * a wrapper of ncclComm_t with MegRay interface
 * collective communications are performed synchronously
 */
class NcclCommunicator : public Communicator {
    public:
        NcclCommunicator(int nranks, int rank);

        ~NcclCommunicator();

        // get a serialized string of ncclUniqueId
        std::string get_uid() override;

        Status init(const std::vector<std::string>& uids) override;

        Status send(const void* sendbuff, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) override;

        Status recv(void* recvbuff, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) override;

        Status scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) override;

        Status gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) override;

        Status all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, std::shared_ptr<Context> ctx) override;

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
        ncclUniqueId m_uid;
        ncclComm_t m_comm;
        bool m_inited;
};

} // namespace MegRay
