/**
 * \file src/core/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megray/communicator.h"

namespace MegRay {

Status Communicator::init(const char* master_ip, int port) {
    m_client = std::make_shared<Client>(m_nranks, m_rank);
    MEGRAY_CHECK(m_client->connect(master_ip, port));
    return do_init();
}

Status Communicator::recv(void* recvbuf, size_t len, DType dtype, uint32_t rank,
                          std::shared_ptr<Context> ctx) {
    size_t type_size = get_dtype_size(dtype);
    return _recv(recvbuf, len * type_size, rank, ctx);
}

Status Communicator::send(const void* sendbuff, size_t len, DType dtype,
                          uint32_t rank, std::shared_ptr<Context> ctx) {
    size_t type_size = get_dtype_size(dtype);
    return _send(sendbuff, len * type_size, rank, ctx);
}

}  // namespace MegRay
