/**
 * \file src/communicator.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "context.h"

namespace MegRay {

/*!
 * abstract class of MegRay main interface
 * MegRay communicator corresponds to a nccl communicator or a ucp worker
 * providing send/recv and collective communication methods
 */
class Communicator {
    public:
        // construct a MegRay communicator with the rank of this process
        // and the number of all ranks
        Communicator(uint32_t nranks, uint32_t rank) : m_nranks(nranks), m_rank(rank) {}

        // get the number of all ranks
        uint32_t nranks() { return m_nranks; }

        // get the rank of this process
        uint32_t rank() { return m_rank; }

        // get the unique id of this communicator
        virtual std::string get_uid() = 0;

        // build a group with unique ids of all communicators in the group
        virtual Status init(const std::vector<std::string>& uids) = 0;

        // send data to another communicator in the group
        virtual Status send(const void* sendbuff, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) = 0;

        // receive data from another communicator in the group
        virtual Status recv(void* recvbuf, size_t len, uint32_t rank,
                std::shared_ptr<Context> ctx) = 0;

        // the length of sendbuff = recvlen * m_nranks
        // the length of recvbuff = recvlen
        virtual Status scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) = 0;

        // the length of sendbuff = sendlen
        // the length of recvbuff = sendlen * m_nranks
        virtual Status gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) = 0;

        // the length of sendbuff = the length of recvbuff = len * m_nranks
        virtual Status all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, std::shared_ptr<Context> ctx) = 0;

	// the length of sendbuff = sendlen
        // the length of recvbuff = sendlen * m_nranks
        virtual Status all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                DType dtype, std::shared_ptr<Context> ctx) = 0;

        // the length of sendbuff = the length of recvbuff = len
        virtual Status all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) = 0;

        // the length of sendbuff = recvlen * m_nranks
        // the length of recvbuff = recvlen
        virtual Status reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) = 0;

	// the length of sendbuff = the length of recvbuff = len
        virtual Status broadcast(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, uint32_t root, std::shared_ptr<Context> ctx) = 0;

	// the length of sendbuff = the length of recvbuff = len
        virtual Status reduce(const void* sendbuff, void* recvbuff, size_t len,
                DType dtype, ReduceOp op, uint32_t root, std::shared_ptr<Context> ctx) = 0;

    protected:
        uint32_t m_nranks;
        uint32_t m_rank;
};

/*!
 * get a communicator implemented with nccl or ucx
 * return std::shared_ptr<NcclCommunicator> or std::shared_ptr<UcxCommunicator>
 */
std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank, Backend backend);

} // namespace MegRay
