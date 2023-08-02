/**
 * \file include/megray.h
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

#include "megray/communicator.h"
#include "megray/config.h"
#include "megray/server.h"

#ifdef MEGRAY_WITH_CUDA
#include "megray/cuda_context.h"
#endif

#ifdef MEGRAY_WITH_HIP
#include "megray/hip_context.h"
#endif

#ifdef MEGRAY_WITH_CNCL
#include "megray/cnrt_context.h"
#endif

namespace MegRay {

/*!
 * get a communicator implemented with nccl or shm
 * return std::shared_ptr<NcclCommunicator> or std::shared_ptr<ShmCommunicator>
 */
std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank,
                                               Backend backend);

}  // namespace MegRay
