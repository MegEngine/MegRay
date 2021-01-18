/**
 * \file src/megray.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megray.h"

#ifdef MEGRAY_WITH_NCCL
#include "nccl/communicator.h"
#endif

#ifdef MEGRAY_WITH_UCX
#include "ucx/communicator.h"
#endif

#ifdef MEGRAY_WITH_RCCL
#include "rccl/communicator.h"
#endif

namespace MegRay {

std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank,
                                               Backend backend) {
    std::shared_ptr<Communicator> comm;
    switch (backend) {
#ifdef MEGRAY_WITH_NCCL
        case MEGRAY_NCCL:
            comm = std::make_shared<NcclCommunicator>(nranks, rank);
            break;
#endif
#ifdef MEGRAY_WITH_UCX
        case MEGRAY_UCX:
            comm = std::make_shared<UcxCommunicator>(nranks, rank);
            break;
#endif
#ifdef MEGRAY_WITH_RCCL
        case MEGRAY_RCCL:
            comm = std::make_shared<RcclCommunicator>(nranks, rank);
            break;
#endif
        default:
            MEGRAY_THROW("unknown backend");
    }
    return comm;
}

}  // namespace MegRay
