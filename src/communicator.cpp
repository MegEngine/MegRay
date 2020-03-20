/**
 * \file src/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "communicator.h"
#include "nccl/communicator.h"
#include "ucx/communicator.h"

namespace MegRay {

std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank, Backend backend) {
    std::shared_ptr<Communicator> comm;
    switch (backend) {
        case MEGRAY_NCCL:
            comm = std::make_shared<NcclCommunicator>(nranks, rank);
            break;
        case MEGRAY_UCX:
            comm = std::make_shared<UcxCommunicator>(nranks, rank);
            break;
        default:
            MEGRAY_THROW("unknown backend");
    }
    return comm;
}

} // namespace MegRay
